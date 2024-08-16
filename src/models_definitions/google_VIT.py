from transformers import AutoModel, AutoImageProcessor
from PIL import Image

IMAGE_PATH = "images/image3.jpg"


def load_vit() -> "tuple[AutoImageProcessor, AutoModel]":
    processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    vit = AutoModel.from_pretrained("google/vit-base-patch16-224-in21k")
    return processor, vit


def main():
    processor, vit = load_vit()
    with Image.open(IMAGE_PATH) as img:
        image_processed = processor(img, return_tensors="pt")
        print(image_processed["pixel_values"].shape)
        outputs = vit(**image_processed)

    print(outputs.last_hidden_state.shape)


if __name__ == "__main__":
    main()
