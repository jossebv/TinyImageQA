import os
from transformers import AutoTokenizer, AutoModelForQuestionAnswering


def main():
    question = "What is my name?"
    context = "My name is Jose and I live in Madrid"

    tokenizer = AutoTokenizer.from_pretrained("Intel/dynamic_tinybert")
    model = AutoModelForQuestionAnswering.from_pretrained("Intel/dynamic_tinybert")
    print(model)

    tokens = tokenizer(question, context, return_tensors="pt")
    resp = model(**tokens)

    answer_start_index = resp.start_logits.argmax()
    answer_end_index = resp.end_logits.argmax()
    predict_answer_tokens = tokens.input_ids[
        0, answer_start_index : answer_end_index + 1
    ]
    print(tokenizer.decode(predict_answer_tokens))


if __name__ == "__main__":
    main()
