import os
import random
import argparse
from datetime import datetime

import torch
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight

from config import config
from src.data_utils import get_datamodule
from src.modeling import get_model


torch.set_float32_matmul_precision('high')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda-device", "-D", type=int, default=0,
                        help="Which GPU to run on, if more than one is available")  # noqa

    subparsers = parser.add_subparsers(dest="command")
    train_parser = subparsers.add_parser("train", help="Run model training")
    train_parser.add_argument(
        "config_file", type=str, help="Path to yaml config file.")

    val_parser = subparsers.add_parser("validate", help="Run model validation")
    val_parser.add_argument(
        "config_file", type=str, help="Path to yaml config file.")
    val_parser.add_argument("--split", type=str, default="validation",
                            choices=["train", "validation", "test"])

    predict_parser = subparsers.add_parser("predict", help="Run prediction")
    predict_parser.add_argument(
        "config_file", type=str, help="Path to yaml config file.")
    predict_parser.add_argument("--split", type=str, default="validation",
                                choices=["train", "validation", "test"])

    return parser.parse_args()


def main(args):
    config.load_yaml(args.config_file)

    start_time = datetime.now()
    print(f"Experiment start: {start_time}")
    print()
    print(config)

    random.seed(config.Experiment.random_seed.value)
    np.random.seed(config.Experiment.random_seed.value)
    torch.manual_seed(config.Experiment.random_seed.value)

    run_kwargs = {}
    if args.command == "train":
        run_fn = run_train
    elif args.command == "validate":
        run_fn = run_validate
        run_kwargs["datasplit"] = args.split
    elif args.command == "predict":
        run_fn = run_validate
        run_kwargs["datasplit"] = args.split
    else:
        raise argparse.ArgumentError(f"Unknown command '{args.command}'.")
    run_fn(config, **run_kwargs)

    end_time = datetime.now()
    print()
    print(f"Experiment end: {end_time}")
    print(f"  Time elapsed: {end_time - start_time}")


def run_train(config):
    outdir = config.Experiment.output_dir.value
    ckpt_dir = os.path.join(outdir, "checkpoint")
    ckpt_file = os.path.join(ckpt_dir, "adapter_model.safetensors")
    if os.path.isfile(ckpt_file):
        raise OSError(f"Trained model already found at {ckpt_file}!")
    os.makedirs(outdir, exist_ok=True)
    config.yaml(outpath=os.path.join(outdir, "config.yaml"))

    datamodule = get_datamodule(config)
    model = get_model(config)
    model.label_weights = datamodule.compute_label_weights(model.vocab_size).to("cuda")  # noqa
    train_loader, val_loader, _ = datamodule.prepare_for_model(model, shuffle_train=True)  # noqa
    model.train(config, train_loader, val_loader)


def run_validate(config, datasplit="validation"):
    config.Training.batch_size.value = 1

    outdir = config.Experiment.output_dir.value
    os.makedirs(outdir, exist_ok=True)

    datamodule = get_datamodule(config)
    model = get_model(config)
    model.label_weights = datamodule.compute_label_weights(model.vocab_size)  # noqa
    iids = datamodule.tokenizer(datamodule.labels, add_special_tokens=False)["input_ids"]  # noqa
    iids = [i for idxs in iids for i in idxs]
    split_loaders = datamodule.prepare_for_model(model, shuffle_train=False)
    datasplit_map = dict(zip(["train", "validation", "test"], split_loaders))
    dataloader = datasplit_map[datasplit]
    outputs = model.validate_and_predict(
            config, dataloader, datamodule.label_ids)

    preds_dir_str = ''
    if config.Model.Lora.load_lora_checkpoint.value is True or config.Model.load_finetuned_checkpoint.value is True:
        preds_dir_str = "finetune"
        if config.Model.Lora.use_lora.value is False:
            preds_dir_str += "_no_lora"
    else:
        preds_dir_str = f"{config.Data.fewshot_examples.value}_shot"
    preds_dir = os.path.join(outdir, "predictions", preds_dir_str)
    os.makedirs(preds_dir, exist_ok=True)
    config.yaml(outpath=os.path.join(preds_dir, "config.yaml"))
    gold_labels = [ex["label"] for ex in datamodule.splits[datasplit]]
    preds_decoded = datamodule.tokenizer.batch_decode(
            outputs["predictions"], skip_special_tokens=True)
    probs = outputs["logits"].softmax(1)
    probs_by_label = {lab: probs[:, i].tolist()
                      for (i, lab) in enumerate(datamodule.labels)}
    out_df = pd.DataFrame({"gold_labels": gold_labels,
                           "predictions": preds_decoded,
                           **probs_by_label})
    out_df.to_csv(os.path.join(preds_dir, f"{datasplit}.csv"), index=False)

    target = torch.LongTensor(
            [datamodule.labels.index(lab) for lab in gold_labels])
    label_weights = compute_class_weight(
            class_weight="balanced",
            y=np.array(gold_labels), classes=np.array(datamodule.labels))
    label_weights = torch.FloatTensor(label_weights).to(probs.device, torch.float)
    loss = torch.nn.functional.nll_loss(
            torch.log(probs.to(torch.float)), target, weight=label_weights)
    loss_file = os.path.join(outdir, "predictions", "losses.csv")
    if os.path.isfile(loss_file):
        losses = pd.read_csv(loss_file)
    else:
        losses = pd.DataFrame()
    new_df = pd.DataFrame({"eval": [preds_dir_str], "loss": [loss.item()]})
    losses = pd.concat([losses, new_df])
    losses.to_csv(loss_file, index=False)


if __name__ == "__main__":
    args = parse_args()
    main(args)
