import os
import random
import pandas as pd
import json
import torch
from torch.utils.data import Dataset
from transformers import default_data_collator
import re  # Import regular expressions
from dataclasses import dataclass, field

DATASET_JSON_FILE = "/opt/llama-cookbook/ts_3gpp_dataset.json"
SOURCE_CODE_DATASET_JSON_FILE = "/opt/llama-cookbook/open5gs_srcode_dataset.json"


def filter_3gppspec_dataset(fiveg_only=True,lte_only=False,common=False):
    dataset_dir = os.path.dirname(DATASET_JSON_FILE)
    fiveg_dataset_file = os.path.join(dataset_dir, "5g_ts_3gpp_dataset.csv")
    lte_dataset_file = os.path.join(dataset_dir, "lte_ts_3gpp_dataset.csv")
    common_dataset_file = os.path.join(dataset_dir, "common_ts_3gpp_dataset.csv")

    if fiveg_only:
        if os.path.exists(fiveg_dataset_file):
            df_5g = pd.read_csv(fiveg_dataset_file) 
            return df_5g
    if lte_only:
        if os.path.exists(lte_dataset_file):
            df_lte = pd.read_csv(lte_dataset_file)
            return df_lte
    if common:
        if os.path.exists(common_dataset_file):
            df_common = pd.read_csv(common_dataset_file)
            return df_common

    # -- Load TS3GPP specification data --
    with open(DATASET_JSON_FILE, "r") as f_spec:
        raw_spec_data = json.load(f_spec)
    df_spec_all = pd.DataFrame(raw_spec_data)
    df_spec_all["type"] = "spec"
    df_spec_all = df_spec_all[df_spec_all["spectype"] == "Technical Specification (TS)"]
    #df_spec = df_spec_all[df_spec_all["technology"].apply(lambda tech_list: any(t in ["5G", "LTE"] for t in tech_list))]
    
    # Convert technology column to tuple for exact matching
    df_spec_all["technology_tuple"] = df_spec_all["technology"].apply(tuple)

    # Get df_5g: Only rows where technology is exactly ["5G"]
    df_5g = df_spec_all[df_spec_all["technology_tuple"] == ("5G",)].drop(columns=["technology_tuple"])
    print("df_5g shape:", df_5g.shape)
    df_5g.to_csv(fiveg_dataset_file, index=False)

    # Get df_lte: Only rows where technology is exactly ["LTE"]
    df_lte = df_spec_all[df_spec_all["technology_tuple"] == ("LTE",)].drop(columns=["technology_tuple"])
    print("df_lte shape:", df_lte.shape)
    df_lte.to_csv(lte_dataset_file, index=False)

    # Get df_common: Rows containing 5G or LTE but not in df_5g or df_lte
    df_common = df_spec_all[
        df_spec_all["technology"].apply(lambda tech: any(t in ["5G", "LTE"] for t in tech))
        & ~df_spec_all.index.isin(df_5g.index)
        & ~df_spec_all.index.isin(df_lte.index)
    ].drop(columns=["technology_tuple"])
    print("df_common shape:", df_common.shape)
    df_common.to_csv(common_dataset_file, index=False)
    
    if fiveg_only: 
        return df_5g
    if lte_only:
        return df_lte
    if common:
        return df_common


class TS3GPPDatasetCPT(Dataset):
    """
    A Dataset wrapping both 3GPP spec items (type='spec') and code items (type='code').
    """
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        item_type = item.get("type", "unknown")  # 'spec' or 'code'
        cpt_text = ""

        if item_type == "spec":
            text = item.get("content", "")
            specnumber = item.get("specnumber", "N/A")
            spec_sec = item.get("title", "No Title")
            cpt_text = f"3GPP TS {specnumber} {spec_sec} : {text}"

        elif item_type == "code":
            filetype = item.get("filetype")
            extention = item.get("extention")
            functionality = item.get("functionality")
            directory = item.get("directory")
            nf = item.get("nf")
            interface = item.get("interface")
            code_content = item.get("content", "")  # assuming code content is stored here

            cpt_text = (
                f"Filetype: {filetype}, Extension: {extention}, "
                f"Functionality: {functionality}, Directory: {directory}, "
                f"Network Function: {nf}, Interface: {interface}\n\n"
                f"Code:\n{code_content}"
            )

        else:
            raise ValueError(f"Unknown item type: {item_type}. Must be 'spec' or 'code'.")
        
        if not isinstance(cpt_text, str):
            cpt_text = str(cpt_text)

        tokenized_output = self.tokenizer(
            cpt_text,
            truncation=True,
            padding="max_length",
            max_length=2048,
            return_tensors="pt"
        )

        input_ids = tokenized_output["input_ids"].squeeze(0).tolist()
        attention_mask = tokenized_output["attention_mask"].squeeze(0).tolist()
        labels = input_ids.copy()  # Copy list
        labels = [-100 if token == self.tokenizer.pad_token_id else token for token in labels]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def split_cpt_data(combined_data, split, test_size_percent=0.2):
    """Splits spec data (list of dictionaries) into train and test sets using slicing."""

    random_seed = 42
    random.seed(random_seed)
    random.shuffle(combined_data)  # Shuffle after setting the seed

    test_size = int(len(combined_data) * test_size_percent)
    train_size = len(combined_data) - test_size
    print(f"Train size: {train_size}, Test size: {test_size}")

    if split == "train":
        #return combined_data[:train_size]
        return combined_data[0:20]
    elif split == "test":
        #return combined_data[train_size:]
        return combined_data[20:40]
    else:
        raise ValueError(f"Invalid split: {split} (must be 'train' or 'test').")


def get_fivegcpt_dataset(dataset_config, tokenizer, split: str):
    """
    Loads and preprocesses the combined dataset (TS3GPP specs + source code) from JSON files.

    Args:
        dataset_config: Dataset configuration dataclass.
        tokenizer: Tokenizer for text inputs.
        split: 'train' or 'test'.

    Returns:
        A torch.utils.data.Dataset object.
    """
    spec_data = []
    code_data = []

    df_spec = filter_3gppspec_dataset(fiveg_only=True,lte_only=False,common=False)
    spec_data = df_spec.to_dict(orient="records")

    if 0:
        # -- Load Source Code data --
        with open(SOURCE_CODE_DATASET_JSON_FILE, "r") as f_code:
            raw_code_data = json.load(f_code)
        code_data = raw_code_data
        for item in code_data:
            item["type"] = "code"

        # Combine them
        combined_data = spec_data + code_data
        random.shuffle(combined_data)

        # Simple 80/20 split
        split_index = int(len(combined_data) * 0.8)
        if split == "train":
            data = combined_data[:split_index]
        elif split == "test":
            data = combined_data[split_index:]
        else:
            raise ValueError(f"Invalid split: {split} (must be 'train' or 'test').")
    
    combined_data = spec_data.copy()
    
    if split == "train":
        data = split_cpt_data(combined_data, split, test_size_percent=0.2)
    elif split == "test":
        data = split_cpt_data(combined_data, split, test_size_percent=0.2)
    else:
        raise ValueError(f"Invalid split: {split} (must be 'train' or 'test').")
    
    print("One entry from {} data:".format(split))
    print(data[0])
    
    dataset = TS3GPPDatasetCPT(
        data=data,
        tokenizer=tokenizer,
    )
    return dataset

