import argparse
import warnings

from config import config
from src.data_utils.util import get_datamodule


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str)
    parser.add_argument("-L", "--max_length", type=int, default=None)
    return parser.parse_args()


def main(args):
    config.load_yaml(args.config_file)
    dm = get_datamodule(config)
    for (split, examples) in dm.splits.items():
        for example in examples:
            encoded = dm.tokenizer(example["prompt"], max_length=None)
            if args.max_length is not None:
                enc_len = len(encoded["input_ids"])
                if enc_len > args.max_length:
                    warnings.warn(f"Input too long! {enc_len}")
                    print(example["prompt"])
                    input()
    print(examples[0]["prompt"])
    print(examples[0]["label"])



if __name__ == "__main__":
    main(parse_args())
