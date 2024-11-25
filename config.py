import os
import re

from experiment_config import Config, get_and_run_config_command
from experiment_config.config import Parameter


config = Config("StanceDetectionConfig")


@config.parameter(group="Experiment", types=str)
def name(val):
    pass


@config.parameter(group="Experiment", default=0, types=int)
def random_seed(val):
    pass


@config.parameter(group="Experiment", default="logs/", types=str)
def logdir(val):
    assert val != ''


@config.parameter(group="Experiment", default=0, types=int)
def version(val):
    assert val >= 0


@config.parameter(group="Data", types=str)
def dataset_name(val):
    assert val != ''


@config.parameter(group="Data", types=str)
def datadir(val):
    assert os.path.isdir(val), "datadir does not exist!"


@config.parameter(group="Data", default=-1, types=int)
def num_examples(val):
    assert (val == -1) or (val >= 1)


@config.parameter(group="Data", default=0, types=int)
def fewshot_examples(val):
    assert val >= 0


@config.parameter(group="Model", default="lorallm", types=str)
def name(val):  # noqa
    assert val != ''


@config.parameter(group="Model", types=str)
def model_path(val):
    assert val != ''


@config.parameter(group="Model", default=False, types=bool)
def load_finetuned_checkpoint(val):
    """
    If True, will try to load from {output_dir}/checkpoint
    """
    pass


@config.parameter(group="Model", default=False, types=bool)
def load_in_kbit(val):
    pass


@config.parameter(group="Model", default=0.0, types=float)
def layer_freeze_percentage(val):
    """
    The percentage of model layers to freeze, sequentially starting from layer 0
    """
    assert 0.0 <= val <= 1.0

@config.parameter(group="Model", default='>', types=str)
def layer_freeze_pattern(val):
    """
    The pattern of model layers to freeze. Following the pattern:
    (\([0-9]+\))?(<|>)(-*)
    An optional layer number in parentheses indicating the first to freeze.
    '>' or '<', indicating freezing forwards or backwards through layers, respectively.
    0 or more '-' indicating the number of layers to skip each time.
    E.g., (2)>- means freeze every other layer in a forward direction starting at layer 2.
    """
    assert re.fullmatch(r"(\([0-9]+\))?(<|>)(-*)", val) is not None

@config.parameter(group="Model.Lora", default=True, types=bool)
def use_lora(val):
    pass


@config.parameter(group="Model.Lora", default=8, types=int)
def rank(val):
    assert val > 0


@config.parameter(group="Model.Lora", default=32, types=int)
def alpha(val):
    assert val > 0


@config.parameter(group="Training", default=1e-4, types=float)
def lr(val):
    assert val > 0


@config.parameter(group="Training", default=4, types=int)
def batch_size(val):
    assert val > 0


@config.parameter(group="Training", default=1, types=int)
def gradient_accumulation_steps(val):
    assert val >= 1


@config.parameter(group="Training", default=10, types=int)
def epochs(val):
    assert val > 0


@config.parameter(group="Training", default=1, types=int)
def eval_every(val):
    assert val > 0


@config.parameter(group="Training", default=0, types=int)
def warmup_steps(val):
    assert val >= 0


@config.parameter(group="Training", default=True, types=bool)
def early_stopping(val):
    pass


@config.on_load
def set_output_dir():
    logdir = config.Experiment.logdir.value
    logdir = os.path.abspath(logdir)
    model_name = os.path.basename(config.Model.model_path.value)
    version = config.Experiment.version.value
    exp_name = config.Experiment.name.value
    seed = config.Experiment.random_seed.value
    version_str = f"version_{version}/seed_{seed}"
    outdir = os.path.join(logdir, exp_name, model_name, version_str)
    if "output_dir" not in config.Experiment:
        config.Experiment.add(Parameter(
            "output_dir", value=outdir, types=str,
            comment="Automatically generated"))
    else:
        config.update("output_dir", outdir, group="Experiment",
                      run_on_load=False)


if __name__ == "__main__":
    get_and_run_config_command(config)
