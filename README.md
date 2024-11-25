# ExU Stance Detection

This repo is for model development as part of the ExU project.

## Requirements

 - [experiment config](https://github.com/jvasilakes/experiment-config)
 - `accelerate`
 - `peft`
 - `pytorch`
 - `sklearn`
 - `transformers`
 - `tqdm`


## Running Experiments

```
$> python config.py new my_config.yaml
```

Edit `my_config.yaml` as you wish for your experiment.


### N-shot (including 0-shot)

```
Data:
  fewshot_examples: N
```

```
python run.py validate --split {train,val} my_config.yaml
```

This will save predictions to `Experiment.output_dir/predictions/Nshot/`.


### Fine-tuning

```
Data:
  fewshot_examples: 0
Model:
  Lora:
    use_lora: true
    load_lora_checkpoint: false
```

```
python run.py train my_config.yaml
```

```
Model:
  Lora:
    load_lora_checkpoint: true
```

```
python run.py validate {train,val} my_config.yaml
```

This will save predictions to `Experiment.output_dir/predictions/finetune/`.


## Evaluation
