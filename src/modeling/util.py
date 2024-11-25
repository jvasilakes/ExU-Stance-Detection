MODEL_REGISTRY = {}


def register_model(name):
    def add_to_registry(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return add_to_registry


def get_model(config):
    model_name = config.Model.name.value
    try:
        model_class = MODEL_REGISTRY[model_name]
    except KeyError:
        raise KeyError(f"Model {model_name} was not found.")
    return model_class.from_config(config)
