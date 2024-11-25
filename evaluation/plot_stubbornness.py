import os
import argparse
from glob import glob

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions_dir", type=str)
    parser.add_argument("--split", type=str,
                        choices=["train", "validation", "test"], default="train")
    parser.add_argument("--plot_title", type=str, default=None)
    parser.add_argument("--split_examples_by", '-S', default=None,
                        choices=[None, "0shot", "labels"],
                        help="If set, plot metrics separately.")
    parser.add_argument("--show_plot", action="store_true", default=False,
                        help="If set, display the plot as well as save it.")
    return parser.parse_args()


def main(args):
    pred_dirs = []
    for member in glob(f"{args.predictions_dir}/*"):
        if os.path.isdir(member) and "0_shot" not in member:
            pred_dirs.append(member)

    zero_shot_file = os.path.join(
            args.predictions_dir, "0_shot", f"{args.split}.csv")
    zero_shot = pd.read_csv(
                zero_shot_file, dtype={"gold_labels": str, "predictions": str})

    all_data = {}
    for pred_dir in pred_dirs:
        preds_file = os.path.join(pred_dir, f"{args.split}.csv")
        stubs_file = os.path.join(pred_dir, f"stubbornness_{args.split}.csv")
        stubs = pd.read_csv(stubs_file)
        preds = pd.read_csv(preds_file)
        preds["zero_shot"] = zero_shot["predictions"]
        # E.g., 0-shot, finetune, etc.
        eval_type = os.path.basename(pred_dir)
        all_data[eval_type] = [stubs, preds]

    for (split_name, split_idxs) in subset_data(preds, split_by=args.split_examples_by):  # noqa
        this_title = args.plot_title
        if split_name is not None:
            this_title = args.plot_title + f" ({split_name})"
        outfile_name = "results"
        if split_name is not None:
            outfile_name += f"_{split_name}"
        outfile_name += f"_{args.split}.pdf"
        outfile = os.path.join(args.predictions_dir, outfile_name)

        fig, axs = plt.subplots(2)
        if args.plot_title is not None:
            fig.suptitle(this_title)
        # Should sort into [*_shot, finetune]
        for eval_type in sorted(all_data.keys()):
            stubs, preds = all_data[eval_type]
            plot_metrics(stubs, preds, axs, eval_type, idxs=split_idxs)
        plt.tight_layout()
        if outfile is not None:
            plt.savefig(outfile, bbox_inches="tight")
        if args.show_plot is True:
            plt.show()


def plot_metrics(stubs, preds, axes, eval_type, idxs=None):
    if idxs is not None:
        stubs = stubs.iloc[idxs]
        preds = preds.iloc[idxs]

    sns.kdeplot(stubs.stubbornness, cut=3, ax=axes[0], alpha=0.5,
                label=eval_type, multiple="stack", clip=(0.0, 1.0))
    sns.kdeplot(stubs.teachability, cut=3, ax=axes[1], alpha=0.5,
                label=eval_type, multiple="stack", clip=(-1.0, 1.0))
    axes[0].set_xlabel("Stubbornness")
    axes[0].set_ylabel('')
    axes[0].set_yticks([])
    axes[0].set_xlim([0.0, 1.0])
    axes[1].set_xlabel("Teachability")
    axes[1].set_ylabel('')
    axes[1].set_yticks([])
    axes[1].set_xlim([-1.0, 1.0])
    axes[1].legend(loc="upper left")


def subset_data(preds, split_by=None):
    if split_by is None:
        return [(None, preds.index)]
    elif split_by == "0shot":
        correct = np.where(preds["zero_shot"] == preds["gold_labels"])[0]
        wrong = np.where(preds["zero_shot"] != preds["gold_labels"])[0]
        return [("0shot_correct", correct), ("0shot_wrong", wrong)]
    elif split_by == "labels":
        labels = [col for col in preds.columns
                  if col not in ["gold_labels", "predictions", "zero_shot"]]
        return [(lab, np.where(preds["gold_labels"] == lab)[0])
                for lab in labels]
    else:
        raise ValueError(f"Unknown split '{split_by}'")


if __name__ == "__main__":
    main(parse_args())
