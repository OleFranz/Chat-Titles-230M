from unsloth import FastLanguageModel
import shutil
import os
import json


script_path = str(os.path.dirname(os.path.realpath(__file__))).replace("\\", "/")
if script_path[-1] != "/": script_path += "/"

if os.path.exists(f"{script_path}Chat-Titles-230M-Merged"):
    shutil.rmtree(f"{script_path}Chat-Titles-230M-Merged")


from transformers import PreTrainedModel, PreTrainedTokenizer

model: PreTrainedModel
tokenizer: PreTrainedTokenizer

DEFAULT_SYSTEM_PROMPT = "You are an assistant that writes short, descriptive titles for chat conversations."
_NAMESPACE_MARKER = 'namespace(system_prompt="", last_user_index=-1)'


def bake_default_system_prompt(chat_template, system_prompt):
    if not chat_template or _NAMESPACE_MARKER not in chat_template:
        raise ValueError("namespace marker not found in chat_template")
    escaped = system_prompt.replace("\\", "\\\\").replace('"', '\\"')
    return chat_template.replace(_NAMESPACE_MARKER, f'namespace(system_prompt="{escaped}", last_user_index=-1)', 1)


def is_baked(chat_template, system_prompt):
    return bool(chat_template) and system_prompt in chat_template


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


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Chat-Titles-230M",
    load_in_4bit=False,
    device_map="auto"
)

if not is_baked(tokenizer.chat_template, DEFAULT_SYSTEM_PROMPT):
    tokenizer.chat_template = bake_default_system_prompt(tokenizer.chat_template, DEFAULT_SYSTEM_PROMPT)

model.save_pretrained_merged(
    "Chat-Titles-230M-Merged",
    tokenizer,
    save_method="merged_16bit"
)

merged_dir = f"{script_path}Chat-Titles-230M-Merged"
write_chat_template(merged_dir, tokenizer.chat_template)


if os.path.exists(f"{script_path}llama.cpp") == False:
    os.system(f"cd {script_path} && git clone https://github.com/ggerganov/llama.cpp")
    os.system(f"cd {script_path} && cmake llama.cpp -B llama.cpp/build -DBUILD_SHARED_LIBS=ON -DGGML_CUDA=OFF -DLLAMA_CURL=OFF")
    os.system(f"cd {script_path} && cmake --build llama.cpp/build --config Release -j --clean-first --target llama-quantize llama-cli llama-gguf-split llama-mtmd-cli")
    os.system(f"cd {script_path} && cp llama.cpp/build/bin/llama-* llama.cpp")

if os.path.exists(f"{script_path}Chat-Titles-230M-GGUF"):
    shutil.rmtree(f"{script_path}Chat-Titles-230M-GGUF")

os.mkdir(f"{script_path}Chat-Titles-230M-GGUF")

os.system(f"cd {script_path} && python llama.cpp/convert_hf_to_gguf.py Chat-Titles-230M-Merged --outfile Chat-Titles-230M-GGUF/Chat-Titles-230M-f16.gguf --outtype f16")

os.system(f"cd {script_path} && .\\llama.cpp\\build\\bin\\Release\\llama-quantize.exe Chat-Titles-230M-GGUF/Chat-Titles-230M-f16.gguf Chat-Titles-230M-GGUF/Chat-Titles-230M-q8_0.gguf q8_0")
os.system(f"cd {script_path} && .\\llama.cpp\\build\\bin\\Release\\llama-quantize.exe Chat-Titles-230M-GGUF/Chat-Titles-230M-f16.gguf Chat-Titles-230M-GGUF/Chat-Titles-230M-q6_k.gguf q6_k")
os.system(f"cd {script_path} && .\\llama.cpp\\build\\bin\\Release\\llama-quantize.exe Chat-Titles-230M-GGUF/Chat-Titles-230M-f16.gguf Chat-Titles-230M-GGUF/Chat-Titles-230M-q4_k_m.gguf q4_k_m")