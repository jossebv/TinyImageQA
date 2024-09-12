import lightning as L
from models_definitions.TinyImageQA import TinyImageQA
from transformers import AutoTokenizer, AutoImageProcessor
import torch
from PIL import Image

CKP_PATH = "TinyImageQA/6pt2sjw3/checkpoints/epoch=19-step=831280.ckpt"
IMG_PATH = "images/image3.jpg"
QUESTION = "Describe the image"


def main():
    model = TinyImageQA.load_from_checkpoint(CKP_PATH)
    image_processor = AutoImageProcessor.from_pretrained(
        "google/vit-base-patch16-224-in21k", use_fast=True
    )
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")

    image = Image.open(IMG_PATH)
    image = image_processor(image.convert("RGB"), return_tensors="pt")[
        "pixel_values"
    ].to("cuda")
    prompt = f"""Your task is to answer a question about an image.<\n>Given the following image features:<\n><\n>Answer the following question:<\n>{QUESTION}"""
    decoder_input_ids = tokenizer(
        "<pad>", add_special_tokens=False, return_tensors="pt"
    ).input_ids.to("cuda")
    prompt_tokens = tokenizer(prompt, return_tensors="pt").to("cuda")

    decoder_input_ids = model.generate_response(
        image,
        prompt_tokens.input_ids.to(torch.int32),
        prompt_tokens.attention_mask,
        decoder_input_ids,
    )

    print(
        f"Generated so far: {tokenizer.decode(decoder_input_ids[0], skip_special_tokens=True)}"
    )
    # print(tokenizer.decode(torch.argmax(response[0], dim=1), skip_special_tokens=True))


if __name__ == "__main__":
    main()
