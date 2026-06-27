from unsloth import FastLanguageModel

title_model, title_tokenizer = FastLanguageModel.from_pretrained(
    model_name="Chat-Titles-230M-Merged",
    load_in_4bit=False,
    device_map="auto"
)

def generate_chat_title(text):
    input_ids = title_tokenizer.apply_chat_template(
        [
            {"role": "user", "content": text},
        ],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    )
    model_device = next(title_model.parameters()).device
    input_ids = input_ids.to(model_device)

    outputs = title_model.generate(
        input_ids=input_ids,
        max_new_tokens=32,
        max_length=None,
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
        top_k=1,
        eos_token_id=title_tokenizer.eos_token_id
    )

    decoded = title_tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
    return decoded.strip()

examples = [
    "Can you summarize the key points from the meeting transcript?",
    "My laptop battery drains fast after the latest update. Any ideas?",
    "Write a short, friendly email to reschedule a call for next week.",
    "I need a workout plan for beginners that I can do at home.",
    "Please explain the difference between HTTP and HTTPS in simple terms."
]

max_length = len(max(examples, key=len))
for example in examples:
    title = generate_chat_title(example)
    print(f"{example}{' ' * (max_length - len(example))} | {title}")

while True:
    text = input("Message: ")
    print("Title:", generate_chat_title(text))