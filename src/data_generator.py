# test.py
import sys
import psutil
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

sys.path.append(".")


def should_save_response():
    print("[Store data]")
    print("Do you want to save this response?")
    key = ""

    while key not in ["y", "n"]:
        print("Press [y]es or [n]o")
        key = input()

    return key == "y"


def save_response(image_path, question, response):
    pass


model = AutoModel.from_pretrained(
    "openbmb/MiniCPM-Llama3-V-2_5-int4", trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(
    "openbmb/MiniCPM-Llama3-V-2_5-int4", trust_remote_code=True
)
model.eval()

image = Image.open("images/image3.jpg").convert("RGB")
question = "What is in the image?"
msgs = [{"role": "user", "content": question}]

print("RAM Used (GB):", psutil.virtual_memory()[3] / 1000000000)

res = model.chat(
    image=image,
    msgs=msgs,
    tokenizer=tokenizer,
    sampling=True,  # if sampling=False, beam_search will be used by default
    temperature=0.7,
    # system_prompt='' # pass system_prompt if needed
)
print(res)
