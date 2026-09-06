from PIL import Image


def analyze_optical_sar(optical_path, sar_path):
    optical = Image.open(optical_path).convert("RGB")
    sar = Image.open(sar_path).convert("L")

    optical_width, optical_height = optical.size
    sar_width, sar_height = sar.size

    optical_mean = sum(optical.convert("L").getdata()) / (
        optical_width * optical_height
    )

    sar_mean = sum(sar.getdata()) / (
        sar_width * sar_height
    )

    if (optical_width, optical_height) == (sar_width, sar_height):
        registration = "The optical and SAR images have matching dimensions."
    else:
        registration = (
            "The optical and SAR images have different dimensions; "
            "co-registration should be performed before final analysis."
        )

    answer = (
        "Optical + SAR analysis completed.\n\n"
        f"Optical image size: {optical_width} × {optical_height}\n"
        f"SAR image size: {sar_width} × {sar_height}\n"
        f"Optical mean intensity: {optical_mean:.2f}\n"
        f"SAR mean intensity: {sar_mean:.2f}\n\n"
        f"{registration}"
    )

    return answer