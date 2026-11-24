"""Small helpers used by the CropAI dashboard (app.py).

They live in their own module so that they can be imported and unit tested
without starting the Streamlit app.
"""
from datetime import date, timedelta

from PIL import Image


def load_uploaded_image(uploaded_file):
    """Open an uploaded image as RGB, or raise ValueError with a message that is safe to show."""
    try:
        return Image.open(uploaded_file).convert("RGB")
    except OSError as error:
        raise ValueError(
            "This file could not be read as an image. Please upload a JPG, PNG or WEBP photo."
        ) from error


def count_low_confidence(confidences_pct, threshold=90.0):
    """Count the confidence scores (in percent) that are below the threshold."""
    return sum(1 for value in confidences_pct if value < threshold)


def calculate_maturity(planting_date, crop_duration):
    """Calculate a planning estimate using planting date and crop duration."""
    duration_days = int(crop_duration)
    maturity_date = planting_date + timedelta(days=duration_days)
    today = date.today()
    elapsed = (today - planting_date).days
    remaining = (maturity_date - today).days
    progress = min(1.0, max(0.0, elapsed / duration_days))

    if progress < 0.15:
        stage = "Early establishment"
    elif progress < 0.60:
        stage = "Vegetative growth"
    elif progress < 0.80:
        stage = "Reproductive growth"
    elif progress < 1.0:
        stage = "Ripening / grain filling"
    else:
        stage = "Expected maturity window reached"

    return {
        "duration_days": duration_days,
        "maturity_date": maturity_date,
        "remaining": remaining,
        "progress": progress,
        "stage": stage,
    }
