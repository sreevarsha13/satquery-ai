
from transformers import BlipProcessor, BlipForQuestionAnswering
from PIL import Image
import torch

MODEL_NAME = "Salesforce/blip-vqa-base"

print("Loading BLIP VQA model...")

processor = BlipProcessor.from_pretrained(MODEL_NAME)
model = BlipForQuestionAnswering.from_pretrained(MODEL_NAME)

model.eval()

print("BLIP VQA model ready!")

def answer_question(image_path, question):
    image = Image.open(image_path).convert("RGB")

    # Process the image separately
    image_inputs = processor.image_processor(
        images=image,
        return_tensors="pt"
    )

    # Process the question separately
    text_inputs = processor.tokenizer(
        question,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=20
    )

    with torch.no_grad():
        output = model.generate(
            pixel_values=image_inputs["pixel_values"],
            input_ids=text_inputs["input_ids"],
            attention_mask=text_inputs["attention_mask"],
            max_length=30
        )

    answer = processor.decode(
        output[0],
        skip_special_tokens=True
    )

    return answer