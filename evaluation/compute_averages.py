import os
import argparse

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("prediction_files", type=str, nargs='+')
    parser.add_argument("--models", type=str, nargs='*')
    parser.add_argument("--subsets", type=str, nargs='*')
    return parser.parse_args()


def main(args):
    all_pred_dfs = []
    for pred_file in args.prediction_files:
        pred_df = pd.read_csv(pred_file, index_col=["model", "subset"])
        all_pred_dfs.append(pred_df)

    models_to_summ = args.models
    if models_to_summ is None:
        models_to_summ = ["0_shot", "fewshot", "finetune"]
    subsets_to_summ = args.subsets
    if subsets_to_summ is None:
        subsets_to_summ = ["macro", "0-shot correct", "0-shot wrong"]

    averages = []
    for model in models_to_summ:
        for subset in subsets_to_summ:
            if model == "0_shot" and "0-shot" in subset:
                continue
            data = pd.DataFrame()
            for pred_df in all_pred_dfs:
                _model = model
                if model == "fewshot":
                    model_types = pred_df.index.get_level_values("model").unique().values
                    _model = [m for m in model_types if m != "0_shot" and "_shot" in m][0]
                try:
                    this_data = pred_df.loc[(_model, subset)]
                except KeyError:
                    continue
                data = pd.concat([data, this_data], axis=1)
            avg_data = data.mean(axis=1)
            avg_data.name = '_'.join((model, subset))
            averages.append(avg_data)
    avg_df = pd.concat(averages, axis=1)
    print(avg_df.T.to_string())



if __name__ == "__main__":
    main(parse_args())
