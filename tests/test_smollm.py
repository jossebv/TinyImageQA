# pip install transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "HuggingFaceTB/SmolLM-360M-Instruct"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint)

message = "Given the following context:\nI have a Raspberry PI 5\nAnswer the following question. Keep the response concise:\nWhat are the possible configurations?\n"

inputs = tokenizer.encode(message, return_tensors="pt")
outputs = model.generate(
    inputs, max_new_tokens=100, temperature=0.6, top_p=0.92, do_sample=True
)
print(tokenizer.decode(outputs[0]))
