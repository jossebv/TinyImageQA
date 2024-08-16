import os
import time
import tensorflow as tf
import pandas as pd
import datasets
from transformers import AutoTokenizer, AutoImageProcessor
from tqdm import tqdm


class DatasetGenerator:
    def __init__(self, save_dir="data", split="train", batch_size=4, num_proc=1):
        os.makedirs(save_dir, exist_ok=True)
        os.environ["HF_DATASETS_CACHE"] = save_dir
        print("Loading dataset...")
        dataset = datasets.load_dataset(
            "openbmb/RLAIF-V-Dataset", split=split, cache_dir=save_dir
        )
        print("Dataset loaded!")

        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        self.image_processor = AutoImageProcessor.from_pretrained(
            "google/vit-base-patch16-224-in21k"
        )

        dataset = dataset.select(range(1000))
        updated_dataset = dataset.map(self.preprocess_function)
        updated_dataset = self.filter_lenghts(updated_dataset)
        self.dataset_tf = updated_dataset.to_tf_dataset(
            batch_size=batch_size,
            columns=["image", "question_ids", "question_attention_mask", "answer_ids"],
            collate_fn=self.collate_fn,
        )

    def get_tf_dataset(self):
        return self.dataset_tf

    def preprocess_function(self, example):
        image = self.image_processor(
            example["image"].convert("RGB"), return_tensors="tf", dtype=tf.float32
        )["pixel_values"]
        question = example["question"]
        prompt = f"""Your task is to answer a question about an image.<\n>Given the following image features:<\n><\n>Answer the following question:<\n>{question}"""
        question_tokens = self.tokenizer(prompt, return_tensors="tf")
        question_ids = question_tokens.input_ids
        question_attention_mask = question_tokens.attention_mask
        answer_ids = self.tokenizer(example["chosen"], return_tensors="tf").input_ids

        example_features = {
            "image": image,
            "question_ids": question_ids,
            "question_attention_mask": question_attention_mask,
            "answer_ids": answer_ids,
        }

        return example_features

    def filter_lenghts(self, dataset):
        question_ids = dataset["question_ids"]
        answer_ids = dataset["answer_ids"]

        select_idxs = []
        for idx in tqdm(range(len(question_ids)), desc="Filtering long inputs"):
            question_len = len(question_ids[idx][0])
            answer_len = len(answer_ids[idx][0])
            if question_len <= 512 and answer_len <= 512:
                select_idxs.append(idx)

        dataset = dataset.select(select_idxs)
        return dataset

    def pad_text(self, text_ids, attention_masks=None, padded_len=512):
        padded_texts = []
        padded_attention_masks = []
        for idx in range(len(text_ids)):
            padding = tf.constant([[0, 0], [0, padded_len - text_ids[idx].shape[1]]])
            padded_text = tf.pad(
                text_ids[idx], padding, mode="CONSTANT", constant_values=-100
            )
            padded_texts.append(padded_text)

            if attention_masks is not None:
                padded_attention_mask = tf.pad(
                    attention_masks[idx], padding, mode="CONSTANT", constant_values=0
                )
                padded_attention_masks.append(padded_attention_mask)

        return (padded_texts, padded_attention_masks)

    def collate_fn(self, examples):
        question_ids = [example["question_ids"] for example in examples]
        question_attention_masks = [
            example["question_attention_mask"] for example in examples
        ]
        answer_ids = [example["answer_ids"] for example in examples]

        padded_questions_ids, padded_questions_attention_masks = self.pad_text(
            question_ids, question_attention_masks
        )
        padded_answer_ids, _ = self.pad_text(answer_ids, None)

        collated_examples = {
            "image": tf.concat([example["image"] for example in examples], axis=0),
            "question_ids": tf.concat(padded_questions_ids, axis=0),
            "question_attention_mask": tf.concat(
                padded_questions_attention_masks, axis=0
            ),
            "answer_ids": tf.concat(padded_answer_ids, axis=0),
        }
        return collated_examples


if __name__ == "__main__":
    dataset = DatasetGenerator().get_tf_dataset()
    print(next(iter(dataset)))
    print("Dataloader works!!")
