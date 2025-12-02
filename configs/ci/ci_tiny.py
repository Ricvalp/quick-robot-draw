from ml_collections import ConfigDict, config_dict


def get_config() -> ConfigDict:

    cfg = ConfigDict

    cfg.model = ConfigDict
    cfg.model.hidden_dim = config_dict.placeholder(int)

    cfg.training = ConfigDict

    cfg.data = ConfigDict

    return cfg
