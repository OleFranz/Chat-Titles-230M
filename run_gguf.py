from llama_cpp import Llama

model = Llama(model_path="./Chat-Titles-230M-GGUF/Chat-Titles-230M-q4_k_m.gguf")

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