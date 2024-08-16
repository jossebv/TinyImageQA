# Load model directly
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


def load_model():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    return tokenizer, model


def main():
    prompt = "Given the following context:\nMy name is Jose and I live in Madrid.\nAnswer the following question:\nWhere do I live?"
    resp = "Madrid"

    tokenizer, model = load_model()
    inputs = tokenizer(prompt, return_tensors="pt")
    response = tokenizer(resp, return_tensors="pt")
    emb_layer = model.shared
    embedding = emb_layer(inputs["input_ids"])
    print(embedding.shape)

    resp = model(
        inputs_embeds=embedding,
        attention_mask=inputs.attention_mask,
        labels=response.input_ids,
    )
    logits = resp.logits
    loss = resp.loss.item()
    print(tokenizer.decode(torch.argmax(logits, dim=2)[0], skip_special_tokens=True))
    print(f"Loss: {loss:.5f}")


if __name__ == "__main__":
    main()
