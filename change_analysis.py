from PIL import Image, ImageChops, ImageEnhance
import numpy as np


def create_change_map(image1_path, image2_path, output_path):
    image1 = Image.open(image1_path).convert("RGB")
    image2 = Image.open(image2_path).convert("RGB")

    # Make both images the same size
    image2 = image2.resize(image1.size)

    # Calculate pixel difference
    difference = ImageChops.difference(image1, image2)

    # Calculate change percentage
    diff_array = np.array(difference)
    changed_pixels = np.any(diff_array > 20, axis=2)

    change_percentage = (
        changed_pixels.sum() / changed_pixels.size
    ) * 100

    # Make differences easier to see
    difference = ImageEnhance.Contrast(difference).enhance(4)

    difference.save(output_path)

    return output_path, round(change_percentage, 2)