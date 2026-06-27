import multiprocessing, sys, os

if os.name == "nt":
    # never allow multiprocessing, it could destroy Python installations on Windows
    if multiprocessing.current_process().name != "MainProcess":
        print("The train.py script tried to spawn a worker process!\nTerminating worker, please fix the code!")
        sys.exit(0)


from unsloth import FastLanguageModel, unsloth_train, is_bfloat16_supported
from unsloth.chat_templates import train_on_responses_only

import torch
import json
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from transformers import PreTrainedModel, PreTrainedTokenizer

import random
random.seed(42)

import warnings
warnings.filterwarnings("ignore")


MODEL_NAME = "unsloth/LFM2.5-230M"
DATASET_NAME = "ogrnz/chat-titles"
OUTPUT_DIR = "Chat-Titles-230M"
MAX_SEQ_LEN = 4096
LOAD_IN_4BIT = False
VAL_SPLIT_RATIO = 0.05

LORA_RANK = 256
LORA_ALPHA = 256
LORA_DROPOUT = 0.0

BATCH_SIZE = 8
GRAD_ACCUM = 8
LR = 0.0001
WARMUP_RATIO = 0.1
NUM_EPOCHS = 1
WEIGHT_DECAY = 0.01

DEFAULT_SYSTEM_PROMPT = "You are an assistant that writes short, descriptive titles for chat conversations."
_NAMESPACE_MARKER = 'namespace(system_prompt="", last_user_index=-1)'


def bake_default_system_prompt(chat_template, system_prompt):
    if not chat_template or _NAMESPACE_MARKER not in chat_template:
        raise ValueError("namespace marker not found in chat_template")
    escaped = system_prompt.replace("\\", "\\\\").replace('"', '\\"')
    return chat_template.replace(_NAMESPACE_MARKER, f'namespace(system_prompt="{escaped}", last_user_index=-1)', 1)


def write_chat_template(directory, chat_template):
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, "chat_template.jinja"), "w", encoding="utf-8") as f:
        f.write(chat_template)
    config_path = os.path.join(directory, "tokenizer_config.json")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        config["chat_template"] = chat_template
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)


model: PreTrainedModel
tokenizer: PreTrainedTokenizer

def format_example(example: dict) -> dict:
    messages = [
        {"role": "user", "content": example["message"]},
        {"role": "assistant", "content": example["title"]},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False
    )
    text = text.replace("<think>\n\n</think>\n\n", "")

    return {"text": text}

dataset_dict = load_dataset(DATASET_NAME)
split = dataset_dict["train"].train_test_split(test_size=VAL_SPLIT_RATIO, seed=42)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LEN,
    dtype=torch.bfloat16 if is_bfloat16_supported() else torch.float16,
    load_in_4bit=LOAD_IN_4BIT
)

tokenizer.chat_template = bake_default_system_prompt(tokenizer.chat_template, DEFAULT_SYSTEM_PROMPT)

model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_RANK,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
    ],
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    random_state=42
)

train_dataset = split["train"].map(
    format_example,
    remove_columns=split["train"].column_names,
    num_proc=1
)
val_dataset = split["test"].map(
    format_example,
    remove_columns=split["test"].column_names,
    num_proc=1
)

train_dataset = train_dataset.filter(lambda x: x["text"] is not None)
val_dataset = val_dataset.filter(lambda x: x["text"] is not None)


trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    args=SFTConfig(
        dataset_num_proc=1,
        max_seq_length=MAX_SEQ_LEN,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        warmup_ratio=WARMUP_RATIO,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LR,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        torch_empty_cache_steps=10,
        logging_steps=1,
        eval_steps=50,
        eval_strategy="steps",
        save_strategy="epoch",
        optim="adamw_8bit",
        weight_decay=WEIGHT_DECAY,
        lr_scheduler_type="cosine",
        seed=42,
        output_dir=OUTPUT_DIR,
        report_to="tensorboard"
    )
)

trainer = train_on_responses_only(
    trainer,
    instruction_part="<|im_start|>user\n",
    response_part="<|im_start|>assistant\n"
)

unsloth_train(trainer)

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
write_chat_template(OUTPUT_DIR, tokenizer.chat_template)

print(f"Training completed. Model saved to '{OUTPUT_DIR}'")