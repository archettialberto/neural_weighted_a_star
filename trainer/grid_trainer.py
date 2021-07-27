from torch.utils.data import DataLoader
from yapt import Trainer

from dataset.grid_dataset import WarcraftDataset


class GridTrainer(Trainer):
    def set_data_loaders(self):
        args = self.args
        kwargs = {
            'num_workers': args.data.num_workers,
            'pin_memory': args.data.pin_memory
        } if args.cuda else {}

        data_loaders = dict()

        if "warcraft" in args.data.dataset_name:
            dataset = WarcraftDataset
        elif "pkmn" in args.data.dataset_name:
            dataset = WarcraftDataset  # same interface
        else:
            raise NotImplementedError("Implement the dataset!")

        train_data = dataset(args.data.path, "train")
        data_loaders['train'] = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, **kwargs)

        val_data = dataset(args.data.path, "val")
        data_loaders['val'] = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, **kwargs)

        if args.data.test_data is not None:
            test_data = dataset(args.data.path, "test")
            data_loaders['test'] = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, **kwargs)

        return data_loaders
