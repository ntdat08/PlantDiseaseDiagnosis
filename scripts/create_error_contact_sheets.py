from pathlib import Path
import math
import textwrap

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


BASE_DIR = Path("evaluation/classification/misclassified")
IMAGE_DIR = BASE_DIR / "images"

CSV_PATH = BASE_DIR / "misclassified_images.csv"

OUTPUTS = [
    {
        "actual": "Tomato___Target_Spot",
        "predicted": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
        "filename": "tomato_target_spot_errors.png",
        "title": "Tomato Target Spot -> Tomato Yellow Leaf Curl Virus"
    },
    {
        "actual": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
        "predicted": "Corn_(maize)___healthy",
        "filename": "corn_cercospora_errors.png",
        "title": "Corn Cercospora -> Corn Healthy"
    }
]

COLUMNS = 5
CELL_WIDTH = 250
CELL_HEIGHT = 205
IMAGE_WIDTH = 220
IMAGE_HEIGHT = 145


def load_font(size):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def format_label(value):
    return value.replace("___", " / ").replace("_", " ").replace(",", "")


def create_contact_sheet(df, config):

    selected = df[
        (df["actual_class"] == config["actual"]) &
        (df["predicted_class"] == config["predicted"])
    ].copy()

    selected = selected.reset_index(drop=True)

    if selected.empty:
        print("No images found:", config["filename"])
        return

    rows = math.ceil(len(selected) / COLUMNS)

    canvas = Image.new(
        "RGB",
        (COLUMNS * CELL_WIDTH, rows * CELL_HEIGHT + 55),
        "white"
    )

    draw = ImageDraw.Draw(canvas)

    title_font = load_font(18)
    caption_font = load_font(11)

    draw.text(
        (15, 12),
        f"{config['title']} | Total: {len(selected)}",
        fill="black",
        font=title_font
    )

    for index, row in selected.iterrows():

        image_path = IMAGE_DIR / row["image_file"]

        with Image.open(image_path) as source:
            image = source.convert("RGB")
            image.thumbnail((IMAGE_WIDTH, IMAGE_HEIGHT))

        column = index % COLUMNS
        line = index // COLUMNS

        x = column * CELL_WIDTH
        y = line * CELL_HEIGHT + 55

        image_x = x + (CELL_WIDTH - image.width) // 2
        image_y = y

        canvas.paste(image, (image_x, image_y))

        actual = format_label(row["actual_class"])
        predicted = format_label(row["predicted_class"])

        confidence = float(row["prediction_confidence"]) * 100

        caption = (
            f"#{index + 1} | {row['image_file']}\n"
            f"Pred confidence: {confidence:.1f}%\n"
            f"True: {actual}\n"
            f"Pred: {predicted}"
        )

        draw.multiline_text(
            (x + 8, y + IMAGE_HEIGHT + 5),
            caption,
            fill="black",
            font=caption_font,
            spacing=2
        )

    output_path = BASE_DIR / config["filename"]

    canvas.save(output_path)

    print("Created:", output_path)
    print("Images:", len(selected))


def main():

    if not CSV_PATH.exists():
        raise FileNotFoundError(CSV_PATH)

    df = pd.read_csv(CSV_PATH)

    print("========== CREATE CONTACT SHEETS ==========")

    for config in OUTPUTS:
        create_contact_sheet(df, config)

    print("\nCONTACT SHEETS COMPLETED!")


if __name__ == "__main__":
    main()
