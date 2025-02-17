#!/usr/bin/env python3
import os
import sys
import torch
import random
from torch import amp
from transformers import AutoTokenizer, LlamaForCausalLM, pipeline
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), '../llama_cookbook/datasets'))
from dataset_5g import load_5g_issues

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

SYSTEM_MSG = {
    "role": "system",
    "content": "You are an AI assistant specialized in diagnosing 5G network issues. Your primary goal is to analyze logs, identify the root cause of issues, propose code-level solutions, and provide validation steps for resolving network anomalies in Open5GS deployments."
}

def setup_testing(model_path):

    tokenizer_path = model_path
    if not os.path.exists(os.path.join(model_path, "tokenizer.json")):
        print(f"Warning: Tokenizer not found in {model_path}. Trying default Llama tokenizer...")
        tokenizer_path = "meta-llama/Llama-3.2-1B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = LlamaForCausalLM.from_pretrained(
        model_path,
        #torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="sdpa",
        use_cache=False,
        low_cpu_mem_usage=True
    )
    
    analyzer = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        #torch_dtype=torch.float16,
        device_map="auto"
    )

    dataset = load_5g_issues(split="train[:5%]")

    return analyzer, dataset


def select_query(dataset):
    """Select a query: either a fixed query (index 0) or a random query."""
    fixed_index = 6
    use_random = False
    
    if use_random:
        random_index = random.randint(0, len(dataset) - 1)
        print(f"Using random query at index: {random_index}")
        return dataset[random_index]
    else:
        print(f"Using fixed query at index: {fixed_index}")
        return dataset[fixed_index]


def test_single_sample(analyzer, query):
    print("query = ", query)
    issue = query["issue"]
    #issue = "Issue Title: Don't left-shift by negative amount, which is UB according to C17\n\nDetails:\nSorry, I introduced undefined behavior in 990abbab2cddafec14c7b1cfaa605ea88b46bd7b. This should fix it.\n\nLogs:\n\n\nComments:\nNo comments available.\n\nRelevant Code:\nFile: uncrustify-rules.cfg\nPath: open5gs_source_code/lib/sbi/support/r17-20230301-openapitools-6.4.0/openapi-generator/uncrustify-rules.cfg\nContent:\nFile: uncrustify-rules.cfg\nConfig Key: align_left_shift\nValue: true\n\nFile: uncrustify-rules.cfg\nPath: open5gs_source_code/lib/sbi/support/r16-20230226-openapitools-6.4.0/openapi-generator/uncrustify-rules.cfg\nContent:\nFile: uncrustify-rules.cfg\nConfig Key: align_left_shift\nValue: true\n\nFile: uncrustify-rules.cfg\nPath: open5gs_source_code/lib/sbi/support/r16-20210629-openapitools-5.2.0/openapi-generator/uncrustify-rules.cfg\nContent:\nFile: uncrustify-rules.cfg\nConfig Key: align_left_shift\nValue: true\n\nFile: milenage.c\nPath: open5gs_source_code/lib/crypt/milenage.c\nContent:\nFile: milenage.c\nFunction: static void ShiftBits(uint8_t r, uint8_t rijndaelInput[16],"
    
    input_text = (
        f"Analyze the following 5G network issue and provide a detailed response:\n"
        f"1. Summarize the reported issue.\n"
        f"2. Identify the root cause based on the logs and description.\n"
        f"3. Propose a solution, including necessary code changes or configuration fixes.\n"
        f"4. Suggest validation steps to confirm the fix.\n\n"
        f"{issue}\n\n"
        f"Provide a clear, structured response in the following format:\n"
        f"1. Problem Summary\n2. Root Cause\n3. Proposed Solution\n4. Validation Steps"
    )
    messages = [SYSTEM_MSG, {"role": "user", "content": input_text}]
    
    outputs = analyzer(
        messages,
        max_new_tokens=150,
        min_new_tokens=100,
        num_beams=4,
        temperature=1,
        top_k=40,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3
    )
    generated_resolution = outputs[0]["generated_text"]
    print("\n--- Generated Resolution ---\n")
    print(generated_resolution)


def test_model():
    model_path = "/opt/llama-cookbook/output"
    print(f"Loading model from: {model_path}")
    
    analyzer, dataset = setup_testing(model_path)
    query = select_query(dataset)
    
    print("\nTesting selected query...\n")
    test_single_sample(analyzer, query)


if __name__ == "__main__":
    test_model()
