from model.feature_extractor.combresnet18_pad import CombResnet18Pad


def get_feature_extractor(name, params):
    feature_extractors = {
        "resnet_pad": CombResnet18Pad,
    }
    if name not in feature_extractors.keys():
        raise NotImplementedError("Feature extractor " + name + " not implemented.")
    return feature_extractors[name](**params)
