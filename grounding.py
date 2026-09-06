from PIL import Image, ImageDraw


def create_grounding_evidence(image_path, output_path, question=""):
    image = Image.open(image_path).convert("RGB")

    width, height = image.size

    question = question.lower()

    # Decide which part of the image to highlight
    if "sky" in question:
        left = int(width * 0.05)
        top = int(height * 0.05)
        right = int(width * 0.95)
        bottom = int(height * 0.40)

        label = "Sky Evidence"

    elif "water" in question or "lake" in question:
        left = int(width * 0.05)
        top = int(height * 0.55)
        right = int(width * 0.95)
        bottom = int(height * 0.90)

        label = "Water Evidence"

    else:
        left = int(width * 0.05)
        top = int(height * 0.40)
        right = int(width * 0.95)
        bottom = int(height * 0.90)

        label = "Grounding Evidence"

    draw = ImageDraw.Draw(image)

    draw.rectangle(
        [left, top, right, bottom],
        outline="red",
        width=6
    )

    draw.text(
        (left + 10, top + 10),
        label,
        fill="red"
    )

    image.save(output_path)

    return output_path