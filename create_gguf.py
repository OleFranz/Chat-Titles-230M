from unsloth import FastLanguageModel
import shutil
import os


script_path = str(os.path.dirname(os.path.realpath(__file__))).replace("\\", "/")
if script_path[-1] != "/": script_path += "/"

if os.path.exists(f"{script_path}Chat-Titles-230M-Merged"):
    shutil.rmtree(f"{script_path}Chat-Titles-230M-Merged")


from transformers import PreTrainedModel, PreTrainedTokenizer

model: PreTrainedModel
tokenizer: PreTrainedTokenizer


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Chat-Titles-230M",
    load_in_4bit=False,
    device_map="auto"
)

model.save_pretrained_merged(
    "Chat-Titles-230M-Merged",
    tokenizer,
    save_method="merged_16bit"
)


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