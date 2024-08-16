from transformers import TFAutoModelForSeq2SeqLM, AutoTokenizer
import tensorflow as tf

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = TFAutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

model.summary()

prompt = "Given the following context:\nMy name is Jose and I live in Madrid.\nAnswer the following question:\nWhere do I live?"
input_ids = tokenizer(prompt, return_tensors="tf").input_ids
labels = tokenizer("", return_tensors="tf").input_ids
outputs = model(input_ids=input_ids, labels=labels)

logits = outputs.logits
print(tokenizer.decode(tf.math.argmax(logits, axis=2)[0], skip_special_tokens=True))
