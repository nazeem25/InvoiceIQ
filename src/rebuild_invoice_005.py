
from pathlib import Path
import hashlib
import shutil

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
IMAGE_DIR = ROOT / "data" / "invoices"
LABELS_PATH = ROOT / "data" / "ground_truth" / "invoice_labels.csv"

TARGET = IMAGE_DIR / "invoice_005.png"
DUPLICATE = IMAGE_DIR / "invoice_015.png"
BACKUP = ROOT / "outputs" / "backups" / "invoice_005_before_reconstruction.png"

# Do not overwrite an image unless the known duplicate is confirmed.
def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

if not TARGET.exists() or not DUPLICATE.exists():
    raise FileNotFoundError("Both invoice_005.png and invoice_015.png must exist.")

if sha256(TARGET) != sha256(DUPLICATE):
    raise RuntimeError(
        "The two images are no longer identical. "
        "Inspect them manually before replacing anything."
    )

labels = pd.read_csv(LABELS_PATH, dtype=str)
rows = labels[labels["image_file"] == "invoice_005.png"]

if len(rows) != 1:
    raise ValueError("Expected exactly one ground-truth row for invoice_005.png.")

row = rows.iloc[0]

expected = {
    "invoice_number": "INV-2026-0005",
    "vendor": "Sunrise Traders",
    "invoice_date": "25-08-2026",
    "item": "Printer paper",
    "quantity": "8",
    "unit_price": "651.47",
    "subtotal": "5211.76",
    "tax": "938.12",
    "grand_total": "6149.88",
}

for field, value in expected.items():
    if str(row[field]).strip() != value:
        raise ValueError(
            f"Ground-truth mismatch for {field}: "
            f"expected {value!r}, found {row[field]!r}"
        )

# Preserve the duplicate image before reconstructing it.
BACKUP.parent.mkdir(parents=True, exist_ok=True)
if not BACKUP.exists():
    shutil.copy2(TARGET, BACKUP)

# Create a high-contrast synthetic invoice.
image = Image.new("RGB", (1000, 720), "white")
draw = ImageDraw.Draw(image)

font_path = Path("C:/Windows/Fonts/arial.ttf")
bold_path = Path("C:/Windows/Fonts/arialbd.ttf")

def load_font(size, bold=False):
    path = bold_path if bold else font_path
    try:
        return ImageFont.truetype(str(path), size)
    except OSError:
        return ImageFont.load_default()

title_font = load_font(30, bold=True)
heading_font = load_font(21, bold=True)
body_font = load_font(23)
footer_font = load_font(16)

draw.rectangle((35, 35, 965, 685), outline="black", width=3)
draw.text((65, 60), "SAMPLE PURCHASE INVOICE", fill="black", font=title_font)
draw.line((65, 110, 935, 110), fill="black", width=2)

draw.text((65, 140), f"Vendor: {expected['vendor']}", fill="black", font=body_font)
draw.text(
    (65, 190),
    f"Invoice No: {expected['invoice_number']}",
    fill="black",
    font=body_font,
)
draw.text(
    (65, 240),
    f"Invoice Date: {expected['invoice_date']}",
    fill="black",
    font=body_font,
)

draw.line((65, 295, 935, 295), fill="black", width=2)
draw.text((65, 315), "ITEM DETAILS", fill="black", font=heading_font)
draw.line((65, 355, 935, 355), fill="black", width=1)

lines = [
    f"Item: {expected['item']}",
    f"Quantity: {expected['quantity']}",
    f"Unit Price: {expected['unit_price']}",
    f"Subtotal: {expected['subtotal']}",
    f"Tax (18%): {expected['tax']}",
    f"Grand Total: {expected['grand_total']}",
]

y = 375
for line in lines:
    draw.text((65, y), line, fill="black", font=body_font)
    y += 42

draw.line((65, 635, 935, 635), fill="black", width=1)
draw.text(
    (65, 648),
    "TRAINING DATA - NOT A REAL INVOICE",
    fill="black",
    font=footer_font,
)

image.save(TARGET, format="PNG")

print("RECONSTRUCTION COMPLETE")
print(f"Replaced: {TARGET}")
print(f"Original duplicate backed up at: {BACKUP}")
print(f"New image dimensions: {image.width} x {image.height}")
print(f"New image SHA-256: {sha256(TARGET)}")
print(f"Invoice number: {expected['invoice_number']}")
