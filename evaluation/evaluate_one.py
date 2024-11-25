import argparse

import pandas as pd
from sklearn.metrics import precision_recall_fscore_support


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("preds_file", type=str)
    return parser.parse_args()


def main(args):
    df = pd.read_csv(args.preds_file)
    p, r, f, _ = precision_recall_fscore_support(df.gold_labels, df.predictions, average="macro")
    results = pd.Series({"Precision": p, "Recall": r, "F1": f})
    print(results)


if __name__ == "__main__":
    main(parse_args())
