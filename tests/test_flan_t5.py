# Load model directly
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel
import torch

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")


for param in model.parameters():
    param.requires_grad = False

msg = "Giving the following context:\nMy name is Jose and I live in Madrid.\nAnswer the following question:\nWhat is my name?"

input_ids = tokenizer(
    "Giving the following context:\nMy name is Jose and I live in Madrid.\nAnswer the following question:\nWhat is my name?",
    return_tensors="pt",
).input_ids
labels = tokenizer("Jose", return_tensors="pt").input_ids

# the forward function automatically creates the correct decoder_input_ids
outputs = model(input_ids=input_ids, labels=labels)
logits = outputs.logits
loss = outputs.loss
print(model)
print(logits.shape)
print(tokenizer.decode(torch.argmax(logits, dim=2)[0], skip_special_tokens=True))
print(loss.item())
