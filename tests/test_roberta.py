# Load model directly
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")
model = AutoModelForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2")

question = "What is my name?"
context = "My name is Jose and I live in Madrid"

inputs = tokenizer(context, question, return_tensors="pt")
resp = model(**inputs)
print(resp)
