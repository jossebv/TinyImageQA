from lightning.pytorch.utilities.parsing import save_hyperparameters
from transformers import AutoModelForSeq2SeqLM, AutoModel
import torch
from torch import nn
import lightning as L
from functools import partial
import sys

sys.path.append(".")
import src.models_definitions.LoRA as lora


class TinyImageQA(L.LightningModule):
    def __init__(
        self,
        lora_r,
        lora_alpha,
        lora_query=True,
        lora_key=False,
        lora_value=True,
        lora_projection=False,
        lora_mlp=False,
        lr=1e-4,
    ):
        super().__init__()
        assign_lora = partial(lora.LinearWithLoRA, rank=lora_r, alpha=lora_alpha)

        self.vit = AutoModel.from_pretrained(
            "google/vit-base-patch16-224-in21k"
        ).train()
        self.projection = nn.Linear(768, 768)
        self.llm = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base").train()
        self.embedding_layer = self.llm.shared.train()

        for param in self.llm.parameters():
            param.requires_grad = False
        for param in self.vit.parameters():
            param.requires_grad = False
        for param in self.embedding_layer.parameters():
            param.requires_grad = False

        for block in self.llm.decoder.block:
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

    def generate_response(self, image, prompt_ids, prompt_att_masks, decoder_input_ids):
        self.vit.eval()
        self.embedding_layer.eval()
        self.llm.eval()

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

        outputs = self.llm(
            inputs_embeds=combined_input,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            return_dict=True,
        )

        encoded_sequence = (outputs.encoder_last_hidden_state,)
        lm_logits = outputs.logits

        # sample last token with highest prob
        next_decoder_input_ids = torch.argmax(lm_logits[:, -1:], axis=-1)
        decoder_input_ids = torch.cat(
            [decoder_input_ids, next_decoder_input_ids], axis=-1
        )

        for _ in range(50):
            lm_logits = self.llm(
                None,
                encoder_outputs=encoded_sequence,
                decoder_input_ids=decoder_input_ids,
                return_dict=True,
            ).logits
            next_decoder_input_ids = torch.argmax(lm_logits[:, -1:], axis=-1)
            decoder_input_ids = torch.cat(
                [decoder_input_ids, next_decoder_input_ids], axis=-1
            )

        return decoder_input_ids

    def training_step(self, batch, batch_idx):
        image, prompt_ids, prompt_att_masks, target_ids = batch
        logits, loss = self(image, prompt_ids, prompt_att_masks, target_ids)

        self.log("train/loss", loss, prog_bar=True, on_step=True)

        # TODO: Log metrics

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer


if __name__ == "__main__":
    import sys
    from transformers import AutoTokenizer

    sys.path.append(".")
    from src.data_module import RLAIFDataModule

    model = TinyImageQA(lora_r=4, lora_alpha=16)
    print(model.llm.decoder)

    train_dataloader = RLAIFDataModule(batch_size=4).train_dataloader()
    batch = next(iter(train_dataloader))
    print([ts.size() for ts in batch])

    output_logits, loss = model(*batch)
    print(loss)

    # tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    # print(
    #     tokenizer.decode(
    #         torch.argmax(output_logits[0], dim=1), skip_special_tokens=True
    #     )
    # )
