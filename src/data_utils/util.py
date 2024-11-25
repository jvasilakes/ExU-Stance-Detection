import string


DATASET_REGISTRY = {}


def register_dataset(name):
    def add_to_registry(cls):
        DATASET_REGISTRY[name] = cls
        return cls
    return add_to_registry


def get_datamodule(config):
    ds_name = config.Data.dataset_name.value
    ds_name = ds_name.replace(string.punctuation, '')
    try:
        ds = DATASET_REGISTRY[ds_name]
    except KeyError:
        raise KeyError(f"Dataset {ds_name} was not found.")
    return ds.from_config(config)


def modify_labels(label_ids, tokenizer):
    label_strs = tokenizer.batch_decode(label_ids)
    versions = {}
    for (lab_ids, lab_str) in zip(label_ids, label_strs):
        mods = [lab_str]
        mods.append(f" {lab_str}")
        mods.append(lab_str.title())
        mods.append(f" {lab_str.title()}")
        mods.append(lab_str.upper())
        mods.append(f" {lab_str.upper()}")
        lab_id_str = '_'.join([str(i) for i in lab_ids])
        versions[lab_id_str] = tokenizer(
                mods, add_special_tokens=False)["input_ids"]
    return versions
