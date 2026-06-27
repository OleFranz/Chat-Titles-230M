from llama_cpp import Llama

model = Llama(
    model_path="./Chat-Titles-230M-GGUF/Chat-Titles-230M-q4_k_m.gguf",
    n_ctx=4096,
    verbose=False
)

examples = [
    "Can you summarize the key points from the meeting transcript?",
    "My laptop battery drains fast after the latest update. Any ideas?",
    "Write a short, friendly email to reschedule a call for next week.",
    "I need a workout plan for beginners that I can do at home.",
    "Please explain the difference between HTTP and HTTPS in simple terms."
]

max_length = len(max(examples, key=len))
for example in examples:
    title = model.create_chat_completion(
        messages=[{"role": "user", "content": example}],
        temperature=0.0,
        top_p=1.0,
        top_k=1
    )["choices"][0]["message"]["content"]
    print(f"{example}{' ' * (max_length - len(example))} | {title}")

while True:
    conversation = [
        {"role": "user", "content": input("Message: ")}
    ]

    result = model.create_chat_completion(
        messages=conversation,
        temperature=0.0,
        top_p=1.0,
        top_k=1
    )

    print("Title:", result["choices"][0]["message"]["content"])