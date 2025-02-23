#!/usr/bin/env python3
from datasets import load_dataset, Features, Value

dataset_file = "/opt/llama-cookbook/merged_finetuning_dataset.json"

def load_5g_issues(split):
    return load_dataset(
        path="json",
        data_files=dataset_file,
        name="open5gs",
        trust_remote_code=True,
        split=split,
        features=Features({
            'issue': Value('string'),
            'resolution': Value('string')
        })
    )


def get_5g_dataset(config, tokenizer, split):
    dataset = load_5g_issues(split)

    prompt = (
        f"Analyze the following 5G network issue and provide a root cause analysis, code-level solution if requires, and validation steps:\n{{issue}}\n---\nResolution:\n"
    )

    def apply_prompt_template(sample):
        return {
            "prompt": prompt.format(issue=sample["issue"]),
            "resolution": sample["resolution"],
        }

    dataset = dataset.map(
        apply_prompt_template,
        remove_columns=list(dataset.features),
        num_proc=64,
        desc="Applying prompt template"
    )

    def tokenize_add_label(sample):
        processed_prompt = tokenizer(
            tokenizer.bos_token + sample["prompt"],
            add_special_tokens=False,
            return_tensors="pt"
        )
        processed_resolution = tokenizer(
            sample["resolution"] + tokenizer.eos_token,
            add_special_tokens=False,
            return_tensors="pt"
        )
        prompt_ids = processed_prompt.input_ids[0].tolist()
        resolution_ids = processed_resolution.input_ids[0].tolist()

        sample = {
            "input_ids": prompt_ids + resolution_ids,
            "attention_mask": [1] * (len(prompt_ids) + len(resolution_ids)),
            "labels": [-100] * len(prompt_ids) + resolution_ids,
        }

        return sample

    dataset = dataset.map(
        tokenize_add_label, 
        remove_columns=list(dataset.features),
        num_proc=64,
        desc="Tokenizing dataset"
    )

    # save_path = "tokenized_dataset"
    # dataset.save_to_disk(save_path)
    # print(f"Dataset saved to {save_path}")

    return dataset
