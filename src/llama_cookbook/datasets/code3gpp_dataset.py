import os
import random
import pandas as pd
import json
import torch
from torch.utils.data import Dataset
from transformers import default_data_collator
import re  # Import regular expressions
from dataclasses import dataclass, field

# --- Define Protocol (FiveG) Feature Vocabularies (CUSTOMIZE THESE!) ---
FIVEG_FEATURE_EMBEDDING_DIM = 64  # Example: your "protocol" embedding dim
CODE_FEATURE_EMBEDDING_DIM = 32   # Example dimension for code features

DATASET_JSON_FILE = "/opt/llama-cookbook/ts_3gpp_dataset.json"
SOURCE_CODE_DATASET_JSON_FILE = "/opt/llama-cookbook/open5gs_srcode_dataset.json"  # Path to your source code dataset


def filter_3gpp_dataset(fiveg_only=True,lte_only=False,common=False):
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


class TS3GPPDataset(Dataset):
    """
    A Dataset wrapping both 3GPP spec items (type='spec') and code items (type='code').
    It returns 'fiveg_feature_indices' for spec items and 'code_feature_indices' for code items.
    """
    def __init__(self, data, tokenizer, fiveg_feature_vocab, code_feature_vocab):
        self.data = data
        self.tokenizer = tokenizer
    
        # 'fiveg_feature_vocab'
        self.fiveg_feature_vocab = fiveg_feature_vocab
        self.code_feature_vocab = code_feature_vocab

        # Lists of feature names
        self.fiveg_feature_names = list(fiveg_feature_vocab.keys())
        self.code_feature_names = list(code_feature_vocab.keys())

        # How many unique “fiveg” (protocol) features in total
        self.fiveg_feature_size = sum(len(v) for v in fiveg_feature_vocab.values())
        # How many unique code features in total
        self.code_feature_size = sum(len(v) for v in code_feature_vocab.values())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item.get("content", "")
        if not isinstance(text, str):
            text = str(text)
        item_type = item.get("type", "unknown")  # 'spec' or 'code'

        tokenized_output = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=2048,
            return_tensors="pt"
        )

        input_ids = tokenized_output["input_ids"].squeeze(0).tolist()
        attention_mask = tokenized_output["attention_mask"].squeeze(0).tolist()
        labels = input_ids.copy()  # Copy list
        labels = [-100 if token == self.tokenizer.pad_token_id else token for token in labels]

        fiveg_feature_indices_list = [-1] * self.fiveg_feature_size
        code_feature_indices_list = [-1] * self.code_feature_size

        if item_type == "spec":
            feature_values = {
                "specnumber": item.get("specnumber"),
                "series": item.get("series"),
                "technology": item.get("technology"),
                "topic": item.get("topic"),
                "specwg": item.get("specwg"),
                "specipr": item.get("specipr"),
            }

            for feat_name in self.fiveg_feature_names:
                feat_val = feature_values.get(feat_name, None)
                if feat_val:
                    if isinstance(feat_val, list):
                         # E.g. a list of tags
                        for val in feat_val:
                            try:
                                idx_ = self.fiveg_feature_vocab[feat_name].index(val)
                                fiveg_feature_indices_list.append(idx_)
                            except ValueError:
                                fiveg_feature_indices_list.append(-1)
                    else:
                        # Single value
                        try:
                            idx_ = self.fiveg_feature_vocab[feat_name].index(feat_val)
                            fiveg_feature_indices_list.append(idx_)
                        except ValueError:
                            fiveg_feature_indices_list.append(-1)
                else:
                    # Missing feature
                    fiveg_feature_indices_list.append(-1)
            
            # Pad or truncate to the known size
            if len(fiveg_feature_indices_list) < self.fiveg_feature_size:
                fiveg_feature_indices_list.extend([-1] * (self.fiveg_feature_size - len(fiveg_feature_indices_list)))
            else:
                fiveg_feature_indices_list = fiveg_feature_indices_list[: self.fiveg_feature_size]

        elif item_type == "code":
            code_feature_values = {
                "filetype": item.get("filetype"),
                "extention": item.get("extention"),
                "functionality": item.get("functionality"),
                "directory": item.get("directory"),
                "nf": item.get("nf"),
                "interface": item.get("interface"),
            }

            for feat_name in self.code_feature_names:
                feat_val = code_feature_values.get(feat_name, None)
                if feat_val:
                    try:
                        idx_ = self.code_feature_vocab[feat_name].index(feat_val)
                        code_feature_indices_list.append(idx_)
                    except ValueError:
                        code_feature_indices_list.append(-1)
                else:
                    code_feature_indices_list.append(-1)

            if len(code_feature_indices_list) < self.code_feature_size:
                code_feature_indices_list.extend([-1] * (self.code_feature_size - len(code_feature_indices_list)))
            else:
                code_feature_indices_list = code_feature_indices_list[: self.code_feature_size]

        else:
            raise ValueError(f"Unknown item type: {item_type}. Must be 'spec' or 'code'.")

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "fiveg_feature_indices": fiveg_feature_indices_list,
            "code_feature_indices": code_feature_indices_list,
        }



class TS3GPPDatasetCPT(Dataset):
    """
    A Dataset wrapping both 3GPP spec items (type='spec') and code items (type='code').
    It returns 'fiveg_feature_indices' for spec items and 'code_feature_indices' for code items.
    """
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
    

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item.get("content", "")
        if not isinstance(text, str):
            text = str(text)
        item_type = item.get("type", "unknown")  # 'spec' or 'code'

        tokenized_output = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=2048,
            return_tensors="pt"
        )

        input_ids = tokenized_output["input_ids"].squeeze(0).tolist()
        attention_mask = tokenized_output["attention_mask"].squeeze(0).tolist()
        labels = input_ids.copy()  # Copy list
        labels = [-100 if token == self.tokenizer.pad_token_id else token for token in labels]

        fiveg_feature_indices_list = [-1] * self.fiveg_feature_size
        code_feature_indices_list = [-1] * self.code_feature_size

        if item_type == "spec":
            feature_values = {
                "specnumber": item.get("specnumber"),
                "series": item.get("series"),
                "technology": item.get("technology"),
                "topic": item.get("topic"),
                "specwg": item.get("specwg"),
                "specipr": item.get("specipr"),
            }

        elif item_type == "code":
            code_feature_values = {
                "filetype": item.get("filetype"),
                "extention": item.get("extention"),
                "functionality": item.get("functionality"),
                "directory": item.get("directory"),
                "nf": item.get("nf"),
                "interface": item.get("interface"),
            }

        else:
            raise ValueError(f"Unknown item type: {item_type}. Must be 'spec' or 'code'.")

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def split_spec_data(combined_data, split, test_size_percent=0.2):
    """Splits spec data (list of dictionaries) into train and test sets using slicing."""

    random_seed = 42
    random.seed(random_seed)
    random.shuffle(combined_data)  # Shuffle after setting the seed

    test_size = int(len(combined_data) * test_size_percent)
    train_size = len(combined_data) - test_size
    print(f"Train size: {train_size}, Test size: {test_size}")

    if split == "train":
        #return combined_data[:train_size]
        return combined_data[40:60]
    elif split == "test":
        #return combined_data[train_size:]
        return combined_data[60:80]
    else:
        raise ValueError(f"Invalid split: {split} (must be 'train' or 'test').")


def get_code3gpp_dataset(dataset_config, tokenizer, split: str):
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

    df_spec = filter_3gpp_dataset(fiveg_only=True,lte_only=False,common=False)
    spec_data = df_spec.to_dict(orient="records")

    if 0:
        # -- Load Source Code data --
        with open(SOURCE_CODE_DATASET_JSON_FILE, "r") as f_code:
            raw_code_data = json.load(f_code)
        code_data = raw_code_data
        for item in code_data:
            item["type"] = "code"

        print("Code feature vocabulary:")
        #print(json.dumps(CODE_FEATURE_VOCAB, indent=4))

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
        data = split_spec_data(combined_data, split, test_size_percent=0.2)
    elif split == "test":
        data = split_spec_data(combined_data, split, test_size_percent=0.2)
    else:
        raise ValueError(f"Invalid split: {split} (must be 'train' or 'test').")
    
    print("One entry from {} data:".format(split))
    print(data[0])
    
    dataset = TS3GPPDataset(
        data=data,
        tokenizer=tokenizer,
    )
    return dataset


def get_code3gpp_data_collator(tokenizer, dataset_config):
    """
    Returns either your custom FivegDataCollatorForLanguageModeling
    or the default data collator, depending on the config.
    """
    if dataset_config.is_fiveg_model:
        print("Using Modified Data Collator")
        from llama_cookbook.FivegModel1 import FivegDataCollatorForLanguageModeling

        # Match collator’s expected argument names
        return FivegDataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,  # For continual pre-training with causal LM
            fiveg_feature_vocab_size=0,  # Will fix below—example or real sum
            code_feature_vocab={}, #CODE_FEATURE_VOCAB,
            fiveg_feature_embedding_dim=FIVEG_FEATURE_EMBEDDING_DIM,
            code_feature_embedding_dim=CODE_FEATURE_EMBEDDING_DIM,
        )
    else:
        print("Using Default Data Collator")
        return default_data_collator


def test_code3gpp(dataset_config):
    from transformers import AutoTokenizer
    # Example usage:
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")  # or your model
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset_config.file = __file__ + ":get_ts3gpp_dataset"

    train_dataset = get_code3gpp_dataset(dataset_config, tokenizer, "train")
    test_dataset = get_code3gpp_dataset(dataset_config, tokenizer, "test")

    print(f"Train dataset length: {len(train_dataset)}")
    print(f"Test dataset length: {len(test_dataset)}")

    # Grab one spec item, one code item
    sample_item_spec = next(item for item in train_dataset if item["fiveg_feature_indices"].nelement() > 0)
    sample_item_code = next(item for item in train_dataset if item["code_feature_indices"].nelement() > 0)

    print("\nSample Spec Item from Dataset:")
    print("Input IDs shape:", sample_item_spec["input_ids"].shape)
    print("Attention Mask shape:", sample_item_spec["attention_mask"].shape)
    print("Labels shape:", sample_item_spec["labels"].shape)
    print("FiveG Feature Indices shape:", sample_item_spec["fiveg_feature_indices"].shape)
    print("FiveG Feature Indices:", sample_item_spec["fiveg_feature_indices"])
    print("Code Feature Indices shape:", sample_item_spec["code_feature_indices"].shape)
    print("Code Feature Indices:", sample_item_spec["code_feature_indices"])

    print("\nSample Code Item from Dataset:")
    print("Input IDs shape:", sample_item_code["input_ids"].shape)
    print("Attention Mask shape:", sample_item_code["attention_mask"].shape)
    print("Labels shape:", sample_item_code["labels"].shape)
    print("FiveG Feature Indices shape:", sample_item_code["fiveg_feature_indices"].shape)
    print("FiveG Feature Indices:", sample_item_code["fiveg_feature_indices"])
    print("Code Feature Indices shape:", sample_item_code["code_feature_indices"].shape)
    print("Code Feature Indices:", sample_item_code["code_feature_indices"])

    # Here we can properly set fiveg_feature_vocab_size based on train_dataset
    # For example, sum over the dict you built:
    from llama_cookbook.FivegModel import FivegDataCollatorForLanguageModeling
    fiveg_feature_vocab_size = train_dataset.fiveg_feature_size

    data_collator = FivegDataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        fiveg_feature_vocab_size=fiveg_feature_vocab_size,
        code_feature_vocab=CODE_FEATURE_VOCAB,
        fiveg_feature_embedding_dim=FIVEG_FEATURE_EMBEDDING_DIM,
        code_feature_embedding_dim=CODE_FEATURE_EMBEDDING_DIM,
    )

    # Create a small batch for testing
    sample_batch = [sample_item_spec, sample_item_code]
    collated_batch = data_collator(sample_batch)

    print("\nCollated Batch (using data collator):")
    print("Collated Input IDs shape:", collated_batch["input_ids"].shape)
    print("Collated Attention Mask shape:", collated_batch["attention_mask"].shape)
    print("Collated Labels shape:", collated_batch["labels"].shape)
    print("Collated FiveG Feature Indices shape:", collated_batch["fiveg_feature_indices"].shape)
    print("Collated Code Feature Indices shape:", collated_batch["code_feature_indices"].shape)