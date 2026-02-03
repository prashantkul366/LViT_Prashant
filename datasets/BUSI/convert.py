import json
import csv

# -------- INPUT / OUTPUT --------
# json_path = r"C:\Users\ve00yn139\OneDrive - YAMAHA MOTOR CO., LTD\Desktop\Unext_skip\LViT_Prashant\datasets\BUSI\train.json"
# csv_path = r"C:\Users\ve00yn139\OneDrive - YAMAHA MOTOR CO., LTD\Desktop\Unext_skip\LViT_Prashant\datasets\BUSI\train_data.csv"

json_path = r"C:\Users\ve00yn139\OneDrive - YAMAHA MOTOR CO., LTD\Desktop\Unext_skip\LViT_Prashant\datasets\BUSI\val.json"
csv_path = r"C:\Users\ve00yn139\OneDrive - YAMAHA MOTOR CO., LTD\Desktop\Unext_skip\LViT_Prashant\datasets\BUSI\val_data.csv"

# -------- LOAD JSON --------
with open(json_path, "r") as f:
    data = json.load(f)

# -------- HELPER FUNCTION --------
def flatten_prompt(p):
    """
    Convert prompt value to a single string.
    Handles string / list / missing values.
    """
    if isinstance(p, list):
        return " | ".join(p)
    elif isinstance(p, str):
        return p
    else:
        return ""

# -------- PREPARE CSV --------
rows = []

for item in data:
    row = {
        "segment_id": item.get("segment_id"),
        "img_name": item.get("img_name"),
        "mask_name": item.get("mask_name"),
        "category": item.get("cat"),
        "sentences_num": item.get("sentences_num"),

        # bbox
        "bbox_x1": item.get("bbox", [None]*4)[0],
        "bbox_y1": item.get("bbox", [None]*4)[1],
        "bbox_x2": item.get("bbox", [None]*4)[2],
        "bbox_y2": item.get("bbox", [None]*4)[3],
    }

    # prompts p0–p6
    prompts = item.get("prompts", {})
    for i in range(7):
        key = f"p{i}"
        row[key] = flatten_prompt(prompts.get(key))

    rows.append(row)

# -------- WRITE CSV --------
fieldnames = rows[0].keys()

with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Saved CSV to {csv_path}")
