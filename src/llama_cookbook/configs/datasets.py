# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass


@dataclass
class samsum_dataset:
    dataset: str =  "samsum_dataset"
    train_split: str = "train"
    test_split: str = "validation"


@dataclass
class grammar_dataset:
    dataset: str = "grammar_dataset"
    train_split: str = "src/llama_cookbook/datasets/grammar_dataset/gtrain_10k.csv"
    test_split: str = "src/llama_cookbook/datasets/grammar_dataset/grammar_validation.csv"


@dataclass
class alpaca_dataset:
    dataset: str = "alpaca_dataset"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = "src/llama_cookbook/datasets/alpaca_data.json"

@dataclass
class custom_dataset:
    dataset: str = "custom_dataset"
    file: str = "getting-started/finetuning/datasets/custom_dataset.py"
    train_split: str = "train"
    test_split: str = "validation"
    data_path: str = ""

@dataclass
class llamaguard_toxicchat_dataset:
    dataset: str = "llamaguard_toxicchat_dataset"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class open5gs_dataset:
    dataset: str = "open5gs_dataset"
    train_split: str = "train[:80%]"
    test_split: str = "train[80%:100%]"

@dataclass
class code3gpp_dataset:
    dataset: str = "code3gpp_dataset"
    train_split: str = "train"
    test_split: str = "test"
    spec_feature_vocab_size: int = 502
    code_feature_vocab_size: int = 37
    is_fiveg_model: bool = True
