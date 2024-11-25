import gc
import os
import re
import warnings

import peft
import torch
import transformers
from tqdm import tqdm
from accelerate import Accelerator
from transformers.models.auto.modeling_auto import (
        MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
        MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES)

from .util import register_model

from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING
# Gemma 2 isn't in the default mapping so we add it here
TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING["gemma2"] = ["q_proj", "v_proj"]  # noqa


def modify_labels(label_ids, tokenizer):
    label_strs = tokenizer.batch_decode(label_ids)
    versions = {}
    for (lab_ids, lab_str) in zip(label_ids, label_strs):
        mods = [lab_str]
        mods.append(f" {lab_str}")
        mods.append(lab_str.title())
        mods.append(f" {lab_str.title()}")
        mods.append(lab_str.upper())
        mods.append(f" {lab_str.upper()}")
        lab_id_str = '_'.join([str(i) for i in lab_ids])
        versions[lab_id_str] = tokenizer(
                mods, add_special_tokens=False)["input_ids"]
    return versions


@register_model("lorallm")
class LoraLLM(object):

    @classmethod
    def from_config(cls, config):
        ckpt_dir = None
        if config.Model.load_finetuned_checkpoint.value is True:
            outdir = config.Experiment.output_dir.value
            ckpt_dir = os.path.join(outdir, "checkpoint")
            ckpt_file = os.path.join(ckpt_dir, "model.safetensors")
            lora_file = os.path.join(ckpt_dir, "adapter_model.safetensors")
            if os.path.isfile(ckpt_file) or os.path.isfile(lora_file):
                print(f"Loading trained model from {ckpt_dir}")
            else:
                raise OSError(f"No finetuned checkpoint found at {ckpt_file}")
        return cls(model_path=config.Model.model_path.value,
                   load_in_kbit=config.Model.load_in_kbit.value,
                   use_lora=config.Model.Lora.use_lora.value,
                   model_checkpoint=ckpt_dir,
                   layer_freeze_percentage=config.Model.layer_freeze_percentage.value,  # noqa
                   layer_freeze_pattern=config.Model.layer_freeze_pattern.value,  # noqa
                   lora_rank=config.Model.Lora.rank.value,
                   lora_alpha=config.Model.Lora.alpha.value)

    def __init__(self, model_path, load_in_kbit=False, use_lora=True,
                 model_checkpoint=None, layer_freeze_percentage=0.0,
                 layer_freeze_pattern='>', lora_rank=8, lora_alpha=32):
        """
        model_path (str): the huggingface name of the base model
        load_in_kbit (bool): whether to use 8bit training (default False)
        use_lora (bool): whether to load the model with LoRA (default True).
        model_checkpoint (str, NoneType): if not None, load a trained model
                                         from the specified checkpoint.
        layer_freeze_percentage (float): value in [0,1] specifying the number of
                                         layers to freeze, sequentially starting
                                         from layer 0.
        layer_freeze_pattern (str): '>' left to right, '<' right to left, '-' skip layer
        lora_rank (int): if use_lora is True, the rank to use (default 8)
        lora_alpha (int): if use_lora is True, the alpha to use (default 32)
        """
        self.model_path = model_path
        self.load_in_kbit = load_in_kbit
        self.use_lora = use_lora
        self.model_checkpoint = model_checkpoint
        self.layer_freeze_percentage = layer_freeze_percentage
        self.layer_freeze_pattern = layer_freeze_pattern
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self._architecture = None

        self.config = transformers.AutoConfig.from_pretrained(self.model_path)
        model_kwargs = {"token": os.environ["HF_TOKEN"]}
        if self.model_checkpoint is not None:
            use_peft = False
            if self.use_lora is True:
                use_peft = True
            self.model_class = self._get_hf_model_class(use_peft=use_peft)   # noqa
            model_kwargs["pretrained_model_name_or_path"] = self.model_checkpoint
        else:
            self.model_class = self._get_hf_model_class(use_peft=False)  # noqa
            model_kwargs["pretrained_model_name_or_path"] = self.model_path
        if self.load_in_kbit is True:
            model_kwargs["quantization_config"] = self.bnb_config
        model_kwargs.update(self._get_model_specific_kwargs())

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                self.model_path, add_eos_token=False)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = self.model_class.from_pretrained(**model_kwargs)
        self.architecture = self.get_architecture()
        if self.use_lora is True:
            if self.model_checkpoint is None:
                print("Loading model with LoRA")
                self.model = peft.get_peft_model(self.model, self.peft_config)
                self.model.print_trainable_parameters()

        self.accelerator = Accelerator()

        self.freeze_layers(self.layer_freeze_percentage, self.layer_freeze_pattern)
        self.model = self.accelerator.prepare(self.model)

        # We use the functional version so label weights can be
        # re-computed after model initialization.
        self.loss_fn = torch.nn.functional.cross_entropy
        self.vocab_size = self.model.lm_head.out_features
        self.label_weights = torch.ones(self.vocab_size)

    @property
    def label_weights(self):
        return self._label_weights

    @label_weights.setter
    def label_weights(self, value):
        self._label_weights = value.to(self.model.device)

    @property
    def peft_config(self):
        task_type = getattr(peft.TaskType, self.architecture)
        return peft.LoraConfig(
                r=self.lora_rank,
                lora_alpha=self.lora_alpha,
                target_modules=None,  # Default for this model
                lora_dropout=0.1,
                task_type=task_type,
                inference_mode=False)

    @property
    def bnb_config(self):
        return transformers.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16)

    def _get_hf_model_class(self, use_peft=False):
        arch = self.config.architectures[0]
        if use_peft is False:
            model_class = getattr(transformers, arch)
        else:
            if "ConditionalGeneration" in arch or "Seq2SeqLM" in arch:
                model_class = getattr(peft, "AutoPeftModelForSeq2SeqLM")
            elif "CausalLM" in arch:
                model_class = getattr(peft, "AutoPeftModelForCausalLM")
            else:
                model_path = self.config._name_or_path
                raise ValueError(f"Could not find AutoModel for {model_path}")
        return model_class

    def _get_model_specific_kwargs(self):
        kwargs = {}
        if self.config.architectures[0] == "Gemma2ForCausalLM":
            kwargs["torch_dtype"] = torch.bfloat16
            kwargs["attn_implementation"] = "flash_attention_2"
        return kwargs

    def get_architecture(self):
        cls_name = self.model.__class__.__name__
        if cls_name in MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES.values():
            return "SEQ_2_SEQ_LM"
        elif "Seq2SeqLM" in cls_name:
            return "SEQ_2_SEQ_LM"
        elif cls_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
            return "CAUSAL_LM"
        elif "CausalLM" in cls_name:
            return "CAUSAL_LM"
        else:
            raise AttributeError(f"Could not determine architecture of model {cls_name}")  # noqa

    def freeze_layers(self, percentage_to_freeze=0.0, pattern='>'):
        if percentage_to_freeze == 0.0:
            return
        for (module_name, module) in self.model.named_modules():
            if module_name.split('.')[-1] == "layers":
                layers = module
        num_layers = len(layers)
        num_to_freeze = round(num_layers * percentage_to_freeze)
        assert num_to_freeze < num_layers
        layer_idxs = self._get_freeze_layer_indices(pattern, num_layers)
        print(f"Freezing {num_to_freeze} layers with pattern {pattern}")
        for i in layer_idxs[:num_to_freeze]:
            for param in layers[i].parameters():
                param.requires_grad = False

    def _get_freeze_layer_indices(self, pattern, n):
        r"""
        (\([0-9]+\))?(<|>)-*")
        # E.g., (12)>--  forwards, skipping two, starting at 12  [12, 15, 18, ..., n, 0, 3, 6, 9]
                > forwards, no skipping,  starting at 0 [0, 1, 2, 3, ..., n]
                (4)> backwards, no skipping, starting at 4 [4, 3, 2, 1, 0, n, n-1, n-2, ..., 5]
        """
        pattern_re = re.compile(r"(\([0-9]+\))?(<|>)(-*)")
        parsed = pattern_re.fullmatch(pattern)
        if parsed is None:
            raise ValueError(f"Invalid pattern '{pattern}'")
        start_idx = parsed.group(1)
        if start_idx is not None:
            start_idx = int(start_idx.strip('()'))
        else:
            start_idx = 0
        direction = parsed.group(2)
        num_skips = len(parsed.group(3))

        idxs = list(range(n))
        if direction == '>':
            ordered = idxs[start_idx:] + idxs[:start_idx]
        elif direction == '<':
            ordered = idxs[start_idx+1:] + idxs[:start_idx+1]
            ordered = ordered[::-1]
        if num_skips > 0:
            ordered = self._apply_skips(ordered, num_skips)
        return ordered

    def _apply_skips(self, idxs, num_skips):
        sections = []
        max_i = len(idxs)
        skip_size = num_skips + 1
        for i in range(num_skips+1):
            if i + skip_size >= max_i:
                max_i = len(idxs) - i
                skip_size = max_i - i - 1
            if skip_size < 1:
                break
            sec = idxs[i:max_i:skip_size]
            sections.append(sec)
        ordered = [i for sec in sections for i in sec]
        return ordered

    def train(self, config, train_loader, val_loader):
        optimizer = torch.optim.AdamW(self.model.parameters(),
                                      lr=config.Training.lr.value)
        num_training_steps = len(train_loader) * config.Training.epochs.value
        lr_scheduler = transformers.get_linear_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=config.Training.warmup_steps.value,
                num_training_steps=num_training_steps)
        train_loader, val_loader, optimizer, lr_scheduler = self.accelerator.prepare(  # noqa: E501 line too long
                train_loader, val_loader, optimizer, lr_scheduler)
        grad_accum_steps = config.Training.gradient_accumulation_steps.value

        self.model.train()
        loss_record = {"train_loss": [], "val_loss": []}
        tolerance = 0
        best_epoch = 0
        stopped_early = False

        val_loss = torch.nan
        desc_template = "epoch: {} | train_loss: {:.3f} | val_loss: {:.3f}"
        pbar = tqdm(range(config.Training.epochs.value))
        for epoch in pbar:
            train_loss = self.train_epoch(
                    train_loader, optimizer, lr_scheduler, grad_accum_steps)
            loss_record["train_loss"].append(train_loss)
            if (epoch + 1) % config.Training.eval_every.value == 0:
                val_outputs = self.validate_and_predict(config, val_loader)
                val_loss = val_outputs["loss"]
                loss_record["val_loss"].append(val_loss)
            pbar.set_description(
                    desc_template.format(epoch, train_loss, val_loss))

            if config.Training.early_stopping.value is True:
                min_loss = min(loss_record["val_loss"][:-1], default=None) or torch.inf  # noqa
                if len(loss_record["val_loss"]) > 0:
                    if loss_record["val_loss"][-1] < min_loss:
                        tolerance = 0
                        self.save_model(config)
                        best_epoch = epoch
                    else:
                        tolerance += 1
                loss_is_zero = torch.isclose(torch.as_tensor(min_loss),
                                             torch.tensor(0.0), atol=1e-5).item()
                if loss_is_zero is True or tolerance == 2:
                    print(f"Stopping training at epoch {best_epoch} w/ val_loss={min_loss:.4f}")  # noqa
                    stopped_early = True
                    break

        if stopped_early is False:
            self.save_model(config)
        torch.cuda.empty_cache()
        gc.collect()

    def train_epoch(self, train_loader, optimizer, lr_scheduler, accum_steps):
        total_loss = 0.
        i = 0
        for batch in tqdm(train_loader, leave=None):
            outputs = self.model(input_ids=batch["input_ids"],
                                 attention_mask=batch["attention_mask"],
                                 labels=batch["labels"])
            labels = batch["labels"]
            logits = outputs.logits
            if self.architecture == "CAUSAL_LM":
                logits, labels = self._shift_for_causal(logits, labels)
            weights = self.label_weights.to(logits.device)
            loss = self.loss_fn(logits.view(-1, logits.size(-1)),
                                labels.view(-1), weight=weights,
                                reduction="mean", ignore_index=-100)
            total_loss += loss.detach().cpu().item()
            self.accelerator.backward(loss)
            # Gradient accumulation
            if ((i + 1) % accum_steps == 0) or ((i + 1) == len(train_loader)):
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            i += 1
        return total_loss / len(train_loader)

    def validate_and_predict(self, config, val_loader, label_ids=None):
        val_loader = self.accelerator.prepare(val_loader)
        self.model.eval()
        total_loss = 0.
        all_label_logits = []
        all_preds = []
        all_labels = []

        if label_ids is not None:
            label_versions = modify_labels(label_ids, self.tokenizer)
            label_ids = [torch.LongTensor(lids).to(self.model.device)
                         for lids in label_ids]

        for batch in tqdm(val_loader, desc="validating: ", leave=None):
            # Unfortunately there is no way to get the logits that were
            # then processed into the generation scores in a single forward
            # pass, so we have to run two different forward passes.
            outputs = self.model(input_ids=batch["input_ids"],
                                 attention_mask=batch["attention_mask"],
                                 labels=batch["labels"])
            # This is only necessary for Mistral, since eos == pad token
            # but it doesn't hurt for the other models.
            prompt_attn_mask = torch.ones_like(batch["prompt"], dtype=torch.int)
            prompt_attn_mask[batch["prompt"] == self.tokenizer.pad_token_id] = 0
            outputs_gen = self.model.generate(
                    input_ids=batch["prompt"], attention_mask=prompt_attn_mask,
                    max_new_tokens=20, pad_token_id=self.tokenizer.pad_token_id,
                    do_sample=False, output_scores=True, return_dict_in_generate=True)  # noqa
            labels = batch["labels"]
            logits = outputs.logits
            if self.architecture == "CAUSAL_LM":
                logits, labels = self._shift_for_causal(logits, labels)
            weights = self.label_weights.to(logits.dtype)
            loss = self.loss_fn(logits.view(-1, logits.size(-1)),
                                labels.view(-1), weight=weights,
                                reduction="mean", ignore_index=-100)
            total_loss += loss.detach().cpu().item()

            # Now we get the logits for each label
            if label_ids is not None:
                # [batch, new_tokens, vocab]
                scores = torch.stack(outputs_gen.scores, dim=1)
                pred_token_ids = scores.argmax(-1)
                # This will be a (batch x label_dim) matrix
                ex_scores = []
                for i in range(batch["input_ids"].size(0)):
                    # First we figure out which index in the output predicts a label  # noqa
                    pred_idx = None
                    for lab_ids in label_ids:
                        lab_id_str = '_'.join([str(lid.item()) for lid in lab_ids])  # noqa
                        for label_version in label_versions[lab_id_str]:
                            if len(label_version) > scores.size(1):
                                # If the label is more tokens than predicted,
                                # it can't be that label
                                continue
                            label_version = torch.LongTensor(label_version)
                            label_version = label_version.to(self.model.device)
                            occurs = torch.isin(pred_token_ids[i], label_version)  # noqa
                            occurs = occurs.argwhere().flatten()
                            if len(occurs) > 0:
                                # We only take the first occurrence
                                # in case it predicts multiple times.
                                pred_idx = occurs[0]
                                break
                    if pred_idx is None:
                        warnings.warn(f"Could not find labels in prediction: {pred_token_ids[i]}")  # noqa
                        pred_idx = 0
                    # Then, we get the scores for each label at that index.
                    label_scores = []
                    for lab_ids in label_ids:
                        best_lscore = -torch.inf
                        # Note that a label can be made up of multiple tokens
                        lab_id_str = '_'.join([str(lid.item()) for lid in lab_ids])  # noqa
                        for label_version in label_versions[lab_id_str]:
                            if len(label_version) > scores.size(1):
                                # If the label is more tokens than predicted,
                                # it can't be that label
                                continue
                            pred_idx_end = min(pred_idx + len(label_version), scores.size(1))  # noqa
                            pred_idxs = torch.arange(pred_idx, pred_idx_end)
                            idxs = torch.arange(len(pred_idxs))
                            lscore = scores[i, pred_idxs][idxs, label_version[:len(idxs)]]  # noqa
                            # So we take the mean score across tokens.
                            lscore = lscore.mean().detach().cpu()
                            if lscore > best_lscore:
                                best_lscore = lscore
                        label_scores.append(best_lscore)
                    # The predicted label is the lab_ids with the highest
                    # score from amongst all its label versions
                    pred_label = label_ids[torch.tensor(label_scores).argmax().item()]  # noqa
                    all_preds.append(pred_label)
                    ex_scores.append(label_scores)

                all_label_logits.extend(ex_scores)
                all_labels.extend(labels.detach().cpu())

        if label_ids is not None:
            all_label_logits = torch.tensor(all_label_logits)
        return {"loss": total_loss / len(val_loader),
                "logits": all_label_logits,
                "predictions": all_preds, "labels": all_labels}

    def _shift_for_causal(self, logits, labels):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        return shift_logits, shift_labels

    def save_model(self, config):
        outdir = config.Experiment.output_dir.value
        outpath = os.path.join(outdir, "checkpoint")
        self.accelerator.wait_for_everyone()
        unwrapped = self.accelerator.unwrap_model(self.model)
        unwrapped.save_pretrained(
                outpath, is_main_process=self.accelerator.is_main_process,
                save_function=self.accelerator.save)
        self.accelerator.wait_for_everyone()
