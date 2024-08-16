import os
import datasets
import lightning as L
from datasets import load_dataset
from transformers import AutoTokenizer, AutoImageProcessor
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import tqdm


def pad_prompts(prompts, padding_labels=False):
    max_len = max(*[prompt.shape[1] for prompt in prompts])

    padded_prompts = []
    attention_masks = []
    for prompt in prompts:
        padding_value = -100 if padding_labels else 0
        padded_prompt = torch.cat(
            [
                prompt,
                padding_value
                * torch.ones(1, max_len - prompt.shape[1], device=prompt.device),
            ],
            dim=1,
        )

        attention_mask = torch.cat(
            [
                torch.ones(1, prompt.shape[1], device=prompt.device),
                torch.zeros(1, max_len - prompt.shape[1], device=prompt.device),
            ],
            dim=1,
        )
        padded_prompts.append(padded_prompt)
        attention_masks.append(attention_mask)

    return padded_prompts, attention_masks


def collate_fn(batch):
    images, questions_ids, answer_ids = zip(*batch)
    padded_questions_ids, questions_attn_masks = pad_prompts(questions_ids)
    padded_responses_ids, _ = pad_prompts(answer_ids, padding_labels=True)

    return (
        torch.cat(images),
        torch.cat(padded_questions_ids).to(torch.int32),
        torch.cat(questions_attn_masks),
        torch.cat(padded_responses_ids).to(torch.int64),
    )


class RLAIFDataset(Dataset):
    def __init__(
        self, save_data_path="data", split="train", max_input_length=512, num_proc=1
    ):
        super().__init__()
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        os.makedirs(save_data_path, exist_ok=True)
        os.environ["HF_DATASETS_CACHE"] = save_data_path
        print("Loading dataset...")
        dataset = datasets.load_dataset(
            "openbmb/RLAIF-V-Dataset", split=split, cache_dir=save_data_path
        )
        print("Dataset loaded")
        self.split = split

        self.image_processor = AutoImageProcessor.from_pretrained(
            "google/vit-base-patch16-224-in21k", use_fast=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")

        # dataset = dataset.select(range(100))
        dataset = dataset.map(
            self._preprocess_fn,
            # input_columns=["image", "question", "chosen"],
            remove_columns=[
                "ds_name",
                "origin_dataset",
                "origin_split",
                "image_path",
                "rejected",
                "question",
                "chosen",
            ],
            num_proc=num_proc,
        ).with_format("torch")
        self.data = self.filter_lenghts(dataset, max_input_length=max_input_length)

    def _preprocess_fn(self, example):
        image = self.image_processor(
            example["image"].convert("RGB"), return_tensors="pt"
        )["pixel_values"]
        question = example["question"]
        prompt = f"""Your task is to answer a question about an image.<\n>Given the following image features:<\n><\n>Answer the following question:<\n>{question}"""
        question_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        answer_ids = self.tokenizer(example["chosen"], return_tensors="pt").input_ids

        example_features = {
            "image": image,
            "question_ids": question_ids,
            "answer_ids": answer_ids,
        }

        return example_features

    def filter_lenghts(self, dataset, max_input_length=512):
        question_ids = dataset["question_ids"]
        answer_ids = dataset["answer_ids"]

        select_idxs = []
        for idx in tqdm(range(len(question_ids)), desc="Filtering long inputs"):
            question_len = question_ids[idx].shape[1]
            answer_len = answer_ids[idx].shape[1]
            if question_len <= max_input_length and answer_len <= max_input_length:
                select_idxs.append(idx)

        dataset = dataset.select(select_idxs)
        return dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = item["image"]
        question_ids = item["question_ids"]
        answer_ids = item["answer_ids"]

        return image, question_ids, answer_ids


class RLAIFDataModule(L.LightningDataModule):
    def __init__(self, save_data_path="data", batch_size=8, num_workers=1, num_proc=1):
        super().__init__()
        self.batch_size = batch_size
        self.save_data_path = save_data_path
        self.num_workers = num_workers
        self.num_proc = num_proc

    def train_dataloader(self):
        return DataLoader(
            RLAIFDataset(
                save_data_path=self.save_data_path,
                split="train",
                num_proc=self.num_proc,
            ),
            batch_size=self.batch_size,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
        )


if __name__ == "__main__":
    dataloader = RLAIFDataModule().train_dataloader()
    print([feature.shape for feature in next(iter(dataloader))])
    print("Datamodule works!")
