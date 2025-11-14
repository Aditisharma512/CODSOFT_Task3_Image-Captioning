# caption_generator.py
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import sys

def generate_caption(image_path: str) -> str:
    # Load model and processor (will download the first time)
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    # Load image and prepare inputs
    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, return_tensors="pt")

    # Generate caption
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python caption_generator.py path/to/image.jpg")
        sys.exit(1)

    image_path = sys.argv[1]
    print("Generating caption. This may download model weights the first time...")
    caption = generate_caption(image_path)
    print("🖼️ Caption:", caption)
