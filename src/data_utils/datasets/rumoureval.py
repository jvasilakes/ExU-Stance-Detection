import os
import re
import random

import pandas as pd
from datasets import Dataset

from .abstract_dataset import AbstractDataset
from ..util import register_dataset


@register_dataset("rumoureval")
class RumourEvalDataset(AbstractDataset):

    @property
    def labels(self):
        return ["agree", "disagree", "query", "comment"]

    @property
    def base_prompt(self):
        return """Given a source tweet and its reply, detect the stance that the reply has towards the source tweet. There are four options: agree, disagree, query and comment. If the reply supports the source tweet, answer with agree; if the reply opposes the source tweet, answer with disagree; if the reply asks for additional evidence in relation to the source tweet, answer with query; if the reply makes their own comment without a clear stance, answer with comment."""  # noqa

    @property
    def output_template(self):
        return "Source tweet: {target} Reply: {text} "

    def load(self):
        train_df = pd.read_csv(os.path.join(self.datadir, "train.csv"),
                               lineterminator='\n')
        train = Dataset.from_pandas(self.preprocess(train_df), split="train")

        dev_df = pd.read_csv(os.path.join(self.datadir, "dev.csv"),
                             lineterminator='\n')
        dev = Dataset.from_pandas(self.preprocess(dev_df), split="validation")

        try:
            test_df = pd.read_csv(os.path.join(self.datadir, "test.csv"),
                                  lineterminator='\n')
            test = Dataset.from_pandas(self.preprocess(test_df), split="test")
        except FileNotFoundError:
            test = None
        return train, dev, test

    def preprocess_tweet(self, text):
        flags = re.MULTILINE | re.DOTALL
        text = re.sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", "HTTPURL",
                      text, flags=flags)
        text = re.sub(r"@\w+", "@USER", text, flags=flags)
        return text

    def preprocess(self, df):
        targets = df["target"].apply(self.preprocess_tweet)
        texts = df["text"].apply(self.preprocess_tweet)
        df["label"] = self.remap_labels(df["label"])
        examples = [{"target": trg, "text": txt, "label": lab}
                    for (trg, txt, lab) in zip(targets, texts, df["label"])]
        prompts = self.get_prompts(examples, df["label"],
                                   fewshot=self.fewshot_examples)
        return pd.DataFrame({"prompt": prompts, "label": df["label"]})

    def remap_labels(self, labels):
        remap = {"support": "agree", "deny": "disagree",
                 "query": "query", "comment": "comment"}
        return labels.map(remap)
