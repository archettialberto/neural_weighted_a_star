from model.dnwa import DNWA
from model.nchb import NodeCEModel
from model.node import NodeModel


def get_model(name, params):
    models = {
        "node": NodeModel,
        "nchb": NodeCEModel,
        "dnwa": DNWA,
    }
    if name not in models.keys():
        raise NotImplementedError("Extractor " + name + " not implemented.")
    if params is not None:
        return models[name](**params)
    return models[name]()
