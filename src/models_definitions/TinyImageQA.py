from transformers import AutoModelForSeq2SeqLM, AutoModel
import torch
from torch import nn
import lightning as L
import LoRA as lora
from functools import partial


class TinyImageQA(L.LightningModule):
    def __init__(self, lr=1e-4):
        super().__init__()

        self.vit = AutoModel.from_pretrained(
            "google/vit-base-patch16-224-in21k"
        ).train()
        self.projection = nn.Linear(768, 768)
        self.llm = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base").train()
        self.embedding_layer = self.llm.shared.train()

        self.save_hyperparameters()

    def _combine_features(
        self, text_embedding, image_projection, prompt_att_masks, blank_token_pos
    ):
        combined_features = torch.cat(
            [
                text_embedding[:, : blank_token_pos + 1, :],
                image_projection,
                text_embedding[:, blank_token_pos + 1 :, :],
            ],
            dim=1,
        )

        batch_size, image_features_len, _ = image_projection.shape
        attention_mask = torch.cat(
            [
                prompt_att_masks[:, : blank_token_pos + 1],
                torch.ones(
                    (batch_size, image_features_len), device=prompt_att_masks.device
                ),
                prompt_att_masks[:, blank_token_pos + 1 :],
            ],
            dim=1,
        )

        return combined_features, attention_mask

    def forward(self, image, prompt_ids, prompt_att_masks, target_ids):
        # When combining, the token of newline has the id:2490
        blank_token_position = (
            (prompt_ids[0] == 2490).nonzero(as_tuple=True)[0][1].item()
        )

        image_features = self.vit(image).last_hidden_state
        image_projection = self.projection(image_features)
        text_embedding = self.embedding_layer(prompt_ids)

        combined_input, attention_mask = self._combine_features(
            text_embedding=text_embedding,
            image_projection=image_projection,
            prompt_att_masks=prompt_att_masks,
            blank_token_pos=blank_token_position,
        )

        output = self.llm(
            inputs_embeds=combined_input,
            attention_mask=attention_mask,
            labels=target_ids,
        )
        return output.logits, output.loss

    def training_step(self, batch, batch_idx):
        image, prompt_ids, prompt_att_masks, target_ids = batch
        logits, loss = self(image, prompt_ids, prompt_att_masks, target_ids)

        self.log("train/loss", loss)

        # TODO: Log metrics

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer


class TinyImageQAWLoRA(L.LightningModule):
    def __init__(
        self,
        lora_r,
        lora_alpha,
        lora_dropout,
        lora_query=True,
        lora_key=False,
        lora_value=True,
        lora_projection=False,
        lora_mlp=False,
        lr=1e-4,
    ):
        super().__init__()

        assign_lora = partial(lora.LinearWithLoRA, rank=lora_r, alpha=lora_alpha)
        self.tinyImage = TinyImageQA(lr=lr)
        for param in self.tinyImage.parameters():
            param.requires_grad = False

        for block in self.tinyImage.llm.decoder.block:
            if lora_query:
                block.layer[0].SelfAttention.q = assign_lora(
                    block.layer[0].SelfAttention.q
                )
                block.layer[1].EncDecAttention.q = assign_lora(
                    block.layer[1].EncDecAttention.q
                )
            if lora_key:
                block.layer[0].SelfAttention.k = assign_lora(
                    block.layer[0].SelfAttention.k
                )
                block.layer[1].EncDecAttention.k = assign_lora(
                    block.layer[1].EncDecAttention.k
                )
            if lora_value:
                block.layer[0].SelfAttention.v = assign_lora(
                    block.layer[0].SelfAttention.v
                )
                block.layer[1].EncDecAttention.v = assign_lora(
                    block.layer[1].EncDecAttention.v
                )
            if lora_projection:
                block.layer[0].SelfAttention.o = assign_lora(
                    block.layer[0].SelfAttention.o
                )
                block.layer[1].EncDecAttention.o = assign_lora(
                    block.layer[1].EncDecAttention.o
                )
            if lora_mlp:
                block.layer[2].DenseReluDense.wi_0 = assign_lora(
                    block.layer[2].DenseReluDense.wi_0
                )
                block.layer[2].DenseReluDense.wi_1 = assign_lora(
                    block.layer[2].DenseReluDense.wi_1
                )
                block.layer[2].DenseReluDense.wo = assign_lora(
                    block.layer[2].DenseReluDense.wo
                )

    def forward(self, image, prompt_ids, prompt_att_masks, target_ids):
        logits, loss = self.tinyImage.forward(
            image, prompt_ids, prompt_att_masks, target_ids
        )
        return logits, loss

    def training_step(self, batch, batch_idx):
        image, prompt_ids, prompt_att_masks, target_ids = batch
        logits, loss = self(image, prompt_ids, prompt_att_masks, target_ids)

        self.log("train/loss", loss)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer


if __name__ == "__main__":
    import sys
    from transformers import AutoTokenizer

    sys.path.append(".")
    from src.data_module import RLAIFDataModule

    model = TinyImageQAWLoRA(lora_r=4, lora_alpha=16, lora_dropout=0.05)
    print(model.tinyImage.llm.decoder)

    train_dataloader = RLAIFDataModule(batch_size=4).train_dataloader()
    batch = next(iter(train_dataloader))

    output_logits, loss = model(*batch)
    print(loss)

    # tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    # print(
    #     tokenizer.decode(
    #         torch.argmax(output_logits[0], dim=1), skip_special_tokens=True
    #     )
    # )
