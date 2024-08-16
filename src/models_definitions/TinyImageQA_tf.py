import tensorflow as tf
import keras
from keras import mixed_precision
from transformers import (
    TFAutoModelForSeq2SeqLM,
    TFAutoModel,
)


class TinyImageQA(keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.vit = TFAutoModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.projection = keras.layers.Dense(512, dtype="float32")
        self.t5 = TFAutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
        self.embedding = self.t5.shared

    def combine_features(
        self, prompt, prompt_attention_mask, image_features, blank_token_position
    ):
        combined_features = tf.concat(
            [
                prompt[:, : blank_token_position + 1, :],
                image_features,
                prompt[:, blank_token_position + 1 :, :],
            ],
            axis=1,
        )

        batch_size, image_features_len, _ = tf.shape(image_features)
        attention_mask = tf.concat(
            [
                prompt_attention_mask[:, : blank_token_position + 1],
                tf.ones([batch_size, image_features_len], dtype=tf.int64),
                prompt_attention_mask[:, blank_token_position + 1 :],
            ],
            axis=1,
        )
        return combined_features, attention_mask

    def call(self, inputs):
        image = inputs["image"]
        question_ids = inputs["question_ids"]
        question_attention_mask = inputs["question_attention_mask"]
        answer_ids = inputs["answer_ids"]

        blank_token_position = tf.get_static_value(
            tf.where(tf.equal(question_ids, 2490))[0, 1]
        )

        image_features = self.vit(image).last_hidden_state
        image_features = self.projection(image_features)

        question_embeddings = self.embedding(question_ids)
        combined_features, attention_mask = self.combine_features(
            question_embeddings,
            question_attention_mask,
            image_features,
            blank_token_position,
        )

        outputs = self.t5(
            inputs_embeds=combined_features,
            attention_mask=attention_mask,
            labels=answer_ids,
        )
        self.add_loss(outputs.loss)
        return outputs.logits


if __name__ == "__main__":
    import sys

    sys.path.append(".")
    from src.data_module_tf import DatasetGenerator

    mixed_precision.set_global_policy("mixed_float16")

    dataset = DatasetGenerator().get_tf_dataset()
    batch = next(iter(dataset))

    model = TinyImageQA()
    model(batch)
