import os
import argparse
from glob import glob

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

import stubbornness_metrics as metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions_dir", type=str)
    parser.add_argument("--split", type=str, choices=["train", "validation", "test"],
                        help="The data split to evaluate.")
    return parser.parse_args()


def main(args):
    zero_shot_file = os.path.join(args.predictions_dir,
                                  "0_shot", f"{args.split}.csv")
    if not os.path.isfile(zero_shot_file):
        raise OSError(f"No 0-shot predictions found at {zero_shot_file}.")
    zero_results = evaluate_predictions(zero_shot_file)

    results_to_save = []
    results_to_save.append(f"### {zero_shot_file}\n{zero_results.to_markdown()}\n")  # noqa
    
    l = [0.0 for _ in range(zero_results.shape[0])]
    empty_df = pd.DataFrame({"stub (mean)": l, "stub (sd)": l,
                             "teach (mean)": l,  "teach (sd)": l},
                             index=zero_results.index)
    all_results = pd.concat([zero_results, empty_df], axis=1)
    all_results.insert(0, "model", "0_shot")

    other_preds_glob = os.path.join(args.predictions_dir,
                                    '*', f"{args.split}.csv")
    other_pred_files = sorted(glob(other_preds_glob))
    for other_pred_file in other_pred_files:
        if other_pred_file == zero_shot_file:
            continue
        zero_df = pd.read_csv(
                zero_shot_file, dtype={"gold_labels": str, "predictions": str})
        zero_correct_idxs = np.where(
                zero_df["gold_labels"] == zero_df["predictions"])[0]
        pred_results = evaluate_predictions(other_pred_file, zero_correct_idxs)
        stub_results, stubs, teachs = evaluate_stubbornness(
                zero_shot_file, other_pred_file)
        comb_results = pd.concat([pred_results, stub_results], axis=1)

        results_to_save.append(f"### {other_pred_file}\n{comb_results.to_markdown()}\n")  # noqa
        model_type = other_pred_file.split(os.sep)[-2]
        comb_results.insert(0, "model", model_type)
        all_results = pd.concat([all_results, comb_results], axis=0)

        stubs_df = pd.DataFrame(
                {"stubbornness": stubs, "teachability": teachs})
        outdir = os.path.dirname(other_pred_file)
        outfile = os.path.join(outdir, f"stubbornness_{args.split}.csv")
        stubs_df.to_csv(outfile, index=False)

    results_file = os.path.join(args.predictions_dir, f"prediction_results_{args.split}.md")
    with open(results_file, 'w') as outF:
        outF.write('\n'.join(results_to_save))
    results_csv = os.path.join(args.predictions_dir, f"prediction_results_{args.split}.csv")
    index = all_results.index
    index.name = "subset"
    all_results = all_results.set_index(["model", index])
    all_results.to_csv(results_csv)


def evaluate_predictions(preds_file, zero_correct_idxs=None):
    df = pd.read_csv(
                preds_file, dtype={"gold_labels": str, "predictions": str})
    labels = df["gold_labels"]
    label_set = sorted(set(labels))
    preds = df["predictions"]
    p, r, f, _ = precision_recall_fscore_support(
            labels, preds, labels=label_set, zero_division=0.0)
    mp, mr, mf, _ = precision_recall_fscore_support(
            labels, preds, labels=label_set,
            average="macro", zero_division=0.0)
    precs = np.append(p, mp)
    recs = np.append(r, mr)
    f1s = np.append(f, mf)
    index = label_set + ["macro"]

    if zero_correct_idxs is not None:
        cp, cr, cf, _ = precision_recall_fscore_support(
                labels[zero_correct_idxs], preds[zero_correct_idxs],
                labels=label_set, average="macro", zero_division=0.0)
        zero_wrong_idxs = [i for i in range(len(labels))
                           if i not in zero_correct_idxs]
        wp, wr, wf, _ = precision_recall_fscore_support(
                labels[zero_wrong_idxs], preds[zero_wrong_idxs],
                labels=label_set, average="macro", zero_division=0.0)
        precs = np.append(precs, [cp, wp])
        recs = np.append(recs, [cr, wr])
        f1s = np.append(f1s, [cf, wf])
        index.extend(["0-shot correct", "0-shot wrong"])

    results_df = pd.DataFrame(
            {"Precision": precs,
             "Recall": recs,
             "F1": f1s},
            index=index)
    return results_df


def evaluate_stubbornness(zero_shot, other_preds):
    zero_df = pd.read_csv(
            zero_shot, dtype={"gold_labels": str, "predictions": str})
    other_df = pd.read_csv(
            other_preds, dtype={"gold_labels": str, "predictions": str})
    label_columns = [c for c in zero_df.columns
                     if c not in ["gold_labels", "predictions"]]
    zero_logits = zero_df[label_columns].to_numpy()
    other_logits = other_df[label_columns].to_numpy()
    onehot = np.zeros((len(zero_df), len(label_columns)))
    for (i, gold_lab) in enumerate(zero_df["gold_labels"]):
        lab_idx = label_columns.index(gold_lab)
        onehot[i, lab_idx] = 1.
    stubs = metrics.stubbornness(zero_logits, other_logits, axis=1, norm=True)
    teachs = metrics.teachability(
            zero_logits, other_logits, onehot, axis=1)

    averages = {"stub (mean)": [], "stub (sd)": [],
                "teach (mean)": [], "teach (sd)": []}
    all_idxs = []
    # averages per label
    for i in range(len(label_columns)):
        all_idxs.append(np.argwhere(onehot[:, i] == 1).flatten())
    # averages for zero-shot correct and incorrect
    all_idxs.append(
            np.where(zero_df["gold_labels"] == zero_df["predictions"])[0])
    all_idxs.append(
            np.where(zero_df["gold_labels"] != zero_df["predictions"])[0])
    for (i, idxs) in enumerate(all_idxs):
        averages["stub (mean)"].append(stubs[idxs].mean())
        averages["stub (sd)"].append(stubs[idxs].std())
        averages["teach (mean)"].append(teachs[idxs].mean())
        averages["teach (sd)"].append(teachs[idxs].std())
        if i == (len(label_columns) - 1):
            averages["stub (mean)"].append(np.mean(averages["stub (mean)"]))
            averages["stub (sd)"].append(np.mean(averages["stub (sd)"]))
            averages["teach (mean)"].append(np.mean(averages["teach (mean)"]))
            averages["teach (sd)"].append(np.mean(averages["teach (sd)"]))
    index = label_columns + ["macro"] + ["0-shot correct", "0-shot wrong"]
    results_df = pd.DataFrame(averages, index=index)

    return results_df, stubs, teachs


if __name__ == "__main__":
    main(parse_args())
