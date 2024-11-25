import random
import warnings
from copy import deepcopy

import torch
import numpy as np
from datasets import DatasetDict
from torch.utils.data import DataLoader
from transformers import (AutoTokenizer, DataCollatorForSeq2Seq,
                          DataCollatorWithPadding)
from sklearn.utils.class_weight import compute_class_weight

from ..util import modify_labels


def is_encoder_decoder(hf_model):
    if hasattr(hf_model, "encoder") and hasattr(hf_model, "decoder"):
        return True
    return False


class AbstractDataset(object):

    @classmethod
    def from_config(cls, config):
        return cls(datadir=config.Data.datadir.value,
                   model_path=config.Model.model_path.value,
                   batch_size=config.Training.batch_size.value,
                   num_examples=config.Data.num_examples.value,
                   fewshot_examples=config.Data.fewshot_examples.value,
                   random_seed=config.Experiment.random_seed.value)

    def __init__(self,
                 datadir,
                 model_path,
                 batch_size,
                 num_examples=-1,
                 fewshot_examples=0,
                 random_seed=0):
        self.datadir = datadir
        self.model_path = model_path
        self.batch_size = batch_size
        self.num_examples = num_examples
        self.fewshot_examples = fewshot_examples
        self.random_seed = random_seed

        # Check right away that these are defined
        self.labels
        self.output_template
        self.base_prompt
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        train, val, test = self.load()
        if self.num_examples > -1:
            if self.num_examples < len(train):
                train = train.select(range(self.num_examples))
            if self.num_examples < len(val):
                val = val.select(range(self.num_examples))
            if test is not None:
                if self.num_examples < len(test):
                    test = test.select(range(self.num_examples))
        split_dict = {"train": train, "validation": val}
        if test is not None:
            split_dict["test"] = test
        self.splits = DatasetDict(split_dict)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # This will fail if the child class does not have labels defined.
        self.label_ids = self.tokenizer(self.labels, add_special_tokens=False)["input_ids"]  # noqa

    def compute_label_weights(self, vocab_size):
        weights = torch.empty(vocab_size).fill_(1e-5)
        labels = [ex["label"] for ex in self.splits["train"]]
        label_names = list(set(labels))
        label_weights = compute_class_weight(
                class_weight="balanced",
                classes=np.array(label_names), y=labels)
        label_ids = self.tokenizer(label_names, add_special_tokens=False)["input_ids"]  # noqa
        modified_lab_ids = modify_labels(label_ids, self.tokenizer)
        for (lab_ids, lab_w) in zip(label_ids, label_weights):
            weights[lab_ids] = lab_w
            lab_ids_str = '_'.join([str(lid) for lid in lab_ids])
            mod_lab_ids = modified_lab_ids[lab_ids_str]
            for mod_ids in mod_lab_ids:
                weights[mod_ids] = lab_w
        weights[self.tokenizer.eos_token_id] = 1.0
        return weights

    @property
    def labels(self):
        raise NotImplementedError("labels property is undefined")

    @property
    def output_template(self):
        raise NotImplementedError("output_template is undefined")

    @property
    def base_prompt(self):
        raise NotImplementedError("base_prompt is undefined")

    def remap_label(self, label):
        """
        Called by self.preprocess()
        Override in child classes if necessary.
        """
        return label

    def load(self):
        """
        Should return train, val, test, which are
        all Huggingface Dataset instances.
        """
        raise NotImplementedError("load() method of this dataset is undefined.")  # noqa

    def encode(self, examples, model_architecture):
        # Extra spaces can cause tokenization issues.
        for key in examples.keys():
            examples[key] = [text.strip() for text in examples[key]]

        max_length = self.tokenizer.model_max_length
        if max_length > 1_000_000:
            # If a model is unbounded, model_max_length will just be a huge int
            max_length = 1024

        if model_architecture == "SEQ_2_SEQ_LM":
            model_inputs = self.tokenizer(
                    examples["prompt"], max_length=max_length,
                    padding="max_length", truncation=True, verbose=False)
            labels = self.tokenizer(
                    examples["label"], max_length=10,
                    padding="max_length", truncation=True
                   )["input_ids"]
            for i in range(len(labels)):
                for j in range(len(labels[i])):
                    if labels[i][j] == 0:
                        labels[i][j] = -100  # ignore index
            model_inputs["labels"] = labels
            # Used for generation
            model_inputs["prompt"] = model_inputs["input_ids"]
        elif model_architecture == "CAUSAL_LM":
            input_text = [prompt + ' ' + lab + self.tokenizer.eos_token
                          for (prompt, lab) in zip(examples["prompt"], examples["label"])]  # noqa
            model_inputs = self.tokenizer(
                    input_text, max_length=max_length, padding="max_length",
                    truncation=True, verbose=False)

            # Used for generation
            model_inputs["prompt"] = self.tokenizer(
                    examples["prompt"], max_length=max_length,
                    padding="max_length", truncation=True, verbose=False
                   )["input_ids"]

            # Ignore everything but the label when computing loss
            labels = deepcopy(model_inputs["input_ids"])
            for i in range(len(labels)):
                # First occurrence of non-pad token
                pad_idx = [0 if tid == self.tokenizer.pad_token_id else 1
                           for tid in labels[i]].index(1)
                # We'll ignore the prompt when fine-tuning
                prompt_len = len([tid for tid in model_inputs["prompt"][i]
                                  if tid != self.tokenizer.pad_token_id])  # noqa
                for j in range(pad_idx + prompt_len):
                    labels[i][j] = -100  # ignore index
            model_inputs["labels"] = labels
        return model_inputs

    def prepare_for_model(self, model, shuffle_train=True):
        with model.accelerator.main_process_first():
            processed = self.splits.map(
                    self.encode,
                    fn_kwargs={"model_architecture": model.architecture},
                    batched=True,
                    batch_size=None,
                    num_proc=16,
                    remove_columns=["prompt", "label"],
                    load_from_cache_file=False,
                    desc="Tokenizing dataset")

        if model.architecture == "SEQ_2_SEQ_LM":
            collate_fn = DataCollatorForSeq2Seq(self.tokenizer)
        elif model.architecture == "CAUSAL_LM":
            collate_fn = DataCollatorWithPadding(self.tokenizer)
        train_loader = DataLoader(
                processed["train"], shuffle=shuffle_train,
                collate_fn=collate_fn, batch_size=self.batch_size,
                pin_memory=True)
        val_loader = DataLoader(
                processed["validation"], shuffle=False, collate_fn=collate_fn,
                batch_size=self.batch_size, pin_memory=True)
        test_loader = None
        if "test" in processed.keys():
            test_loader = DataLoader(
                    processed["test"], shuffle=False, collate_fn=collate_fn,
                    batch_size=self.batch_size, pin_memory=True)
        return train_loader, val_loader, test_loader

    def preprocess(self, examples):
        """
        To be called by self.load() in the child class.
        """
        labels = []
        for ex in examples:
            label = self.remap_label(str(ex["label"]))
            labels.append(label)
            ex["label"] = label
        prompts = self.get_prompts(examples, labels,
                                   fewshot=self.fewshot_examples)
        return [{"prompt": prompt, "label": label}
                for (prompt, label) in zip(prompts, labels)]

    def get_prompts(self, examples, labels, fewshot=0):
        prompts = []
        for (i, ex) in enumerate(examples):
            main_example = self.output_template.format(**ex)
            fewshot_text = ''
            if fewshot > 0:
                fewshot_text = self.get_fewshot(
                        main_example, examples, labels, fewshot)
            output = "\nNow complete the following example.\n" + main_example
            prompts.append(self.base_prompt + fewshot_text + output)
        return prompts

    def get_fewshot(self, main_example_text, examples, labels, fewshot_n):
        main_example_enc = self.tokenizer(' '.join(
            ["Now complete the following example.",
             self.base_prompt, main_example_text])
            )
        total_example_len = len(main_example_enc["input_ids"])

        output = "\n### Examples:\n"
        seen_labels = []
        idxs = list(range(len(labels)))
        random.shuffle(idxs)
        for j in idxs[:100]:  # limit the number to look at
            fs_ex = examples[j]
            add_example = False
            if len(set(seen_labels)) == len(self.labels):
                if len(seen_labels) == fewshot_n:
                    break
                else:
                    add_example = True
            elif fs_ex["label"] not in seen_labels:
                add_example = True

            if add_example is True:
                fs_output = self.output_template.format(**fs_ex)
                fs_output += fs_ex["label"] + '\n'
                fs_enc = self.tokenizer(fs_output)
                fs_len = len(fs_enc["input_ids"])
                if fs_len + total_example_len > self.tokenizer.model_max_length:  # noqa
                    continue
                seen_labels.append(fs_ex["label"])
                output += fs_output
                total_example_len += fs_len
        if len(seen_labels) < fewshot_n:
            warnings.warn(f"Only added {len(seen_labels)} fewshot examples.")
        return output
