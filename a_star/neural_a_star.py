import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function

from functools import wraps
from itertools import product

from a_star.a_star_solver import AStarSolver
from a_star.neighbor_utils import NeighborUtils
from utils.collections import AStarSolution


class NeuralAStarSolver(AStarSolver):
    def __init__(self, tau_val):
        super().__init__()
        self.solver = NeuralAStar(tau_val)
        self.nu = None

    def run(self, weights, heuristic, source, target) -> AStarSolution:
        if heuristic is None:
            if self.nu is None:
                self.nu = NeighborUtils(weights.shape[1], weights.shape[2])
            heuristic = self.nu.get_euclidean_heuristic(weights, target)
        return self.solver(weights, heuristic, source, target)


def identity_grad(op):
    """
    Overrides the backward step of a function operating on a tensor with the
    gradient of the identity transformation. I.e., the function is applied on
    the forward step, but ignored on the backward step.

    :param op: A function that takes a tensor as input, and returns a new
        tensor having the same shape
    :return: A function that performs the same operation of "op" in the forward
        step, but which skips it (i.e., propagates the gradients untouched) in
        the backward step
    """
    class PassThrough(Function):

        @staticmethod
        def forward(ctx, x):
            return op(x)

        @staticmethod
        def backward(ctx, op_grad):
            return op_grad

    @wraps(op)
    def wrapped(x):
        return PassThrough.apply(x)

    return wrapped


class BinaryArgmin(nn.Module):
    """
    Implements a differentiable argmin as described in "Path Planning using
    Neural A* Search" Yonetani et al., Equation 6. Note here the input is
    X = G+H, or specifically:
    $$\sigma\left(\frac{\exp (-X / \tau) \odot O}
                       {\langle\exp (-X / \tau), O\rangle}\right)$$
    """

    def __init__(self, tau):
        super().__init__()
        self.tau = tau

    @staticmethod
    def sigma(x):
        """
        Implements the forward step of \sigma() in Equation 6.
        Note: the backward step is overridden with the identity operation (as
        the function is wrapped using the identity_grad decorator)
        :param torch.Tensor x: a tensor of shape [B, N, M]
        :return: a binary tensor of shape [B, N, M]
        """

        B, N, M = x.shape
        device = x.device

        mask = torch.zeros([B, M*N], device=device, dtype=torch.float)
        batch_idx = torch.arange(B, device=device, dtype=torch.int64)
        max_idx = torch.argmax(x.reshape(B, -1), dim=-1)  # [B]

        mask[batch_idx, max_idx] = 1
        mask = mask.reshape(B, N, M)

        return mask

    def forward(self, x, o):
        """
        :param torch.Tensor x: a tensor of shape [B, N, M]
        :param torch.Tensor o: a tensor of shape [B, N, M]
        :return: a tensor of shape [B, N, M]
        """

        x = torch.exp(-x / self.tau) * o  # [B, N, M]
        x = x / x.sum(dim=(1, 2), keepdim=True)  # [B, N, M]
        x_sigma = self.sigma(x)
        x = (x_sigma - x).detach() + x

        return x


class NeuralAStar(nn.Module):
    """
    Implements the algorithm described in "Path Planning using Neural A* Search"
    Yonetani et al.
    """

    def __init__(self, tau):
        super().__init__()

        self.tau = tau
        self.argmin = BinaryArgmin(tau)
        self.register_buffer("kappa", torch.tensor([[[[1, 1, 1],
                                                      [1, 0, 1],
                                                      [1, 1, 1]]]],
                                                   dtype=torch.float))
        self.register_buffer("kcosts",
                             self.costs_kernel())

    def costs_kernel(self, device='cpu'):
        """
        Computes a kernel that can be used to pass from [B, N, M, 8] costs,
        back to [B, N, M] ones. Note that this only works if the tensor on which
        this kernel is convolved only has one pixel with non-zero neighbor costs
        :return: a [1, 8, 3, 3] tensor
        """

        kernel = torch.zeros([1, 8, 3, 3], device=device, dtype=torch.float)
        neigh_idx = list(product((2, 1, 0), (2, 1, 0)))
        neigh_idx.remove((1, 1))
        neigh_idx = torch.tensor(neigh_idx, device=device)  # [8, 2]
        channel_idx = torch.arange(8, device=device)  # [8]

        kernel[:, channel_idx, neigh_idx[:, 0], neigh_idx[:, 1]] = 1

        return kernel

    def neighbor_costs(self, P, Vstar):
        """
        Computes a tensor of shape [B, N, M] containing either 0, if the node
        (i,j) is not a neighbor of the ones selected in Vstar, or the cost of
        going from a node in Vstar to (i,j)
        :param P: a tensor of either shape [B, N, M] or [B, 8, N, M]
        :return: a [B, N, M] tensor
        """

        if P.ndim == 3:
            return P  # [B, N, M]
        else:
            # [B, 1, N, M] * [B, 8, N, M]
            masked_P = Vstar[:, None, :, :] * P
            simple_P = F.conv2d(masked_P, self.kcosts,
                                stride=1, padding=1)  # [B, 1, N, M]
            simple_P = simple_P.squeeze(dim=1)  # [B, N, M]

            return simple_P

    def kconv(self, x):
        """
        :param torch.Tensor x: a tensor of shape [B, N, M]
        :return: a tensor of shape [B, N, M]
        """

        x = x.unsqueeze(dim=1)  # [B, 1, N, M]
        x = F.conv2d(x, self.kappa, stride=1, padding=1)  # [B, 1, N, M]
        x = x.squeeze(dim=1)  # [B, N, M]

        return x

    def update_parents(self, parents, eta, Vnbr, Vstar):
        """
        Updates the parents index tensor
        :param torch.Tensor eta: a
        :param torch.Tensor parents: a [B, N*M] tensor of indices
        :param torch.Tensor Vnbr: a [B, N, M] tensor
        :param torch.Tensor Vstar: a [B, N, M] tensor
        :return: None
        """

        for b in torch.nonzero(eta, as_tuple=False):
            nbr_idx = torch.nonzero(Vnbr[b].reshape(-1), as_tuple=False)
            vstar_idx = torch.nonzero(Vstar[b].reshape(-1), as_tuple=False)
            parents[b, nbr_idx] = vstar_idx

    def backtrack_parents(self, S, vg, shape):
        """
        Extract the shortest path from each sample by backtracking S
        :param torch.Tensor S: a [B, N*M] tensor of indices
        :param torch.Tensor vg: a [B, 2] tensor
        :param Tuple[int] shape: the input shape as a tuple (B, N, M)
        :return: a tuple (shortest_path, shortest_path_expanded) containing two
            tensors of shapes [B, N, M] and [B, N, M, 8] respectively
        """

        B, N, M = shape
        device = S.device
        offsets = list(product((-1, 0, 1), (-1, 0, 1)))
        offsets.remove((0, 0))
        offset_to_chan = dict(zip(offsets, range(8)))

        vg_flat = vg[:, 0] * M + vg[:, 1]
        sp = S.new_zeros([B, N, M, 8]).float()

        for b in range(B):
            curr = vg_flat[b]
            prev_y, prev_x = (-1, -1)
            while curr >= 0:
                curr_y = (curr // N).item()
                curr_x = (curr % N).item()
                if (prev_y, prev_x) != (-1, -1):
                    offset = (prev_y - curr_y, prev_x - curr_x)
                    sp[b, curr_y, curr_x, offset_to_chan[offset]] = 1.0
                curr = S[b, curr]
                prev_y, prev_x = (curr_y, curr_x)

        idx_batch = torch.arange(B, device=device)
        sp_simple = sp.max(dim=-1).values
        sp_simple[idx_batch, vg[:, 0], vg[:, 1]] = 1.0

        return sp_simple, sp

    def forward(self, P, H, vs, vg, X=None):
        """
        Computes the expanded nodes and shortest paths as described in
        Algorithm 2

        :param torch.Tensor P: the psi tensor. The shape of the tensor can be
            either [B, N, M], and in this case the value in position (i,j) is
            interpreted as the cost of passing through (i,j), or [B, N, M, 8],
            and in this case the value (i,j,n) is interpreted as the cost of
            going from (i,j) to the n-th neighbor of (i,j). Neighbors are
            ordered as: [NW, N, NW, W, E, SW, S, SE]
        :param torch.Tensor H: the heuristic tensor of shape [B, N, M]
        :param torch.Tensor X: the x tensor of shape [B, N, M]
        :param torch.Tensor vs: the coordinates of the source node as a (y, x)
            tensor (with 0<=y<N and 0<=x<M) of shape [B, 2]
        :param torch.Tensor vg: the coordinates of the goal node as a (y, x)
            tensor (with 0<=y<N and 0<=x<M) of shape [B, 2]
        :return: a NeuralAStarSolution namedtuple
        """

        Pshape = P.shape if P.ndim == 3 else P.shape[:-1]
        assert vs.ndim == 2 and vs.shape[-1] == 2
        assert Pshape == H.shape and vs.shape == vg.shape
        assert X is None or X.shape == Pshape

        if P.ndim == 4:
            P = P.permute(0, 3, 1, 2)  # [B, 8, N, M]
        else:
            assert P.ndim == 3

        B, N, M = H.shape
        device = H.device
        batch_idx = torch.arange(B, device=device, dtype=torch.int64)

        O = P.new_zeros([B, N, M])
        O[batch_idx, vs[:, 0], vs[:, 1]] = 1

        Vg = P.new_zeros([B, N, M])
        Vg[batch_idx, vg[:, 0], vg[:, 1]] = 1

        C = P.new_zeros([B, N, M])
        G = P.new_zeros([B, N, M])

        eta = P.new_ones(B)

        parents = P.new_zeros([B, N*M], dtype=torch.int64).fill_(-1)

        while torch.nonzero(eta, as_tuple=False).numel() > 0:
            Vstar = self.argmin(G+H, O.detach())
            eta = 1 - (Vg * Vstar).sum(dim=(1, 2)).detach()
            C = C + eta[:, None, None] * Vstar
            O = O - eta[:, None, None] * Vstar
            Vnbr = (self.kconv(Vstar) * (1 - O) * (1 - C)
                    if X is None
                    else self.kconv(Vstar) * (1 - O) * (1 - C) * X)#.detach()
            O = O + Vnbr
            G = (((G * Vstar).sum(dim=(1, 2), keepdim=True).detach()
                  + self.neighbor_costs(P, Vstar.detach())
                  ) * Vnbr.detach()
                 + (G * (1 - Vnbr)).detach())

            self.update_parents(parents, eta, Vnbr, Vstar)

        C = C + Vg
        O = O - Vg

        sp, sp_exp = self.backtrack_parents(parents, vg, (B, N, M))

        return AStarSolution(
            paths=sp,
            exp_nodes=C,
        )
