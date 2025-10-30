# train_grpo.py
#
# See https://github.com/willccbb/verifiers for ongoing developments
#
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["WANDB_DISABLED"] = "true"

import re
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
# from peft import LoraConfig
from IDRR_data import IDRRDataFrames
from trl import GRPOConfig, GRPOTrainer

# Load and prep dataset

SYSTEM_PROMPT = """
Respond in the following format:

<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""
SYSTEM_PROMPT = r"You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \boxed{}. Example format: <think> ... </think> \boxed{A}."

XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""

def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def extract_box_answer(text: str) -> str | None:
    pattern = r"boxed{(.*?)}"
    match = re.search(pattern, text)
    if match:
        return match.group(1).strip()
    return None

def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip().replace(",", "").replace("$", "")

# uncomment middle messages for 1-shot prompting
def get_gsm8k_questions(split = "train") -> Dataset:
    data = load_dataset('openai/gsm8k', 'main')[split] # type: ignore
    data = data.map(lambda x: { # type: ignore
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            #{'role': 'user', 'content': 'What is the largest single-digit prime number?'},
            #{'role': 'assistant', 'content': XML_COT_FORMAT.format(
            #    reasoning="9 is divisble by 3 and 8 is divisible by 2, but 7 is prime.",
            #    answer="7"
            #)},
            {'role': 'user', 'content': x['question']}
        ],
        'answer': extract_hash_answer(x['answer'])
    }) # type: ignore
    return data # type: ignore

def get_pdtb_questions(split = "train") -> Dataset:
    dfs = IDRRDataFrames(
            data_name="pdtb2",
            data_level="top",
            data_relation="Implicit",
            data_path="data/raw/pdtb2.p1.csv",
        )
    answer_map = {
        "Comparison": "A",
        "Contingency": "B",
        "Expansion": "C",
        "Temporal": "D"
    }
    with open("prompts/baseline.txt", 'r') as f:
        prompt_template = f.read()
    
    data = Dataset.from_pandas(dfs.train_df)
    data = data.map(lambda x: {
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': prompt_template.replace("{arg1}", x['arg1']).replace("{arg2}", x['arg2'])}
        ],
        'answer': answer_map[x['label11']]
    }).select_columns(['prompt', 'answer'])
    return data

# dataset = get_gsm8k_questions()
dataset = get_pdtb_questions()
print('-'*10 + " Sample Data " + '-'*10)
print(dataset[0])
print('-'*30)

# Reward functions
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    # extracted_responses = [extract_xml_answer(r) for r in responses]
    extracted_responses = [extract_box_answer(r) for r in responses]
    print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    # pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    pattern = r"^<think>\n.*?\n</think>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses] 
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    # pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    pattern = r"<think>.*?</think>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses] 
    return [0.5 if match else 0.0 for match in matches]

def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1])*0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]

def boxed_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"boxed\{([^}]*)\}"
    responses = [completion[0]["content"] for completion in completions]
    scores = []
    for r in responses:
        score = 0.0
        matches = re.findall(pattern, r)
        if matches:
            score += 0.3
            if matches[-1] in ['A', 'B', 'C', 'D']:
                score += 0.2
        scores.append(score)
    return scores


# model_name = "../pretrained_models/Qwen/Qwen3-1.7B"
model_name = "../pretrained_models/Qwen/Qwen2.5-1.5B-Instruct"
run_name = "qwen2.5-1.5b-grpo-pdtb2"
# run_name = "qwen3-1.7b-grpo-pdtb2"
output_dir = f"./expts/{run_name}"
    
training_args = GRPOConfig(
    output_dir=output_dir,
    run_name=run_name,
    learning_rate=5e-6,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type='cosine',
    logging_steps=1,
    bf16=True,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    num_generations=8,
    max_prompt_length=256,
    max_completion_length=786,
    num_train_epochs=1,
    save_steps=100,
    max_grad_norm=0.1,
    report_to="swanlab",
    log_on_each_node=False,
)
# peft_config = LoraConfig(
#     r=16,
#     lora_alpha=64,
#     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
#     task_type="CAUSAL_LM",
#     lora_dropout=0.05,
# )
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map=None
).to("cuda")
        
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# use peft at your own risk; not working for me with multi-GPU training
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        # xmlcount_reward_func,
        # soft_format_reward_func,
        # strict_format_reward_func,
        # int_reward_func,
        boxed_format_reward_func,
        correctness_reward_func],
    args=training_args,
    train_dataset=dataset,
    #peft_config=peft_config
)
trainer.train()