"""Stage 2: label images from their folder name (no=0, yes=1) and split
90% train+validation / 10% test (stratified). Saves train/val/test CSVs."""
import json
import os

import pandas as pd
from sklearn.model_selection import train_test_split

from pipeline.common import read_params

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")


def main():
    p = read_params()
    root = os.path.join(p["data_dir"], "dataset")

    rows = []
    for cls in p["classes"]:
        folder = os.path.join(root, cls)
        rows += [(os.path.join(folder, f), cls) for f in sorted(os.listdir(folder))
                 if f.lower().endswith(IMG_EXT) and not f.startswith(".")]
    df = (pd.DataFrame(rows, columns=["image_filepaths", "tumor_class"])
            .sample(frac=1, random_state=p["seed"]).reset_index(drop=True))
    if df.tumor_class.nunique() < 2:
        raise SystemExit("Need images in both yes/ and no/ folders")

    trainval, test = train_test_split(df, test_size=p["test_size"], random_state=p["seed"],
                                      stratify=df.tumor_class)
    train, val = train_test_split(trainval, test_size=p["val_size"], random_state=p["seed"],
                                  stratify=trainval.tumor_class)

    out = "artifacts/preprocessing"
    os.makedirs(out, exist_ok=True)
    summary = {}
    for name, part in [("train", train), ("val", val), ("test", test)]:
        part.to_csv(f"{out}/{name}.csv", index=False)
        summary[name] = {"total": len(part), **part.tumor_class.value_counts().to_dict()}
        print(name, summary[name])
    with open(f"{out}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)



if __name__ == "__main__":
    main()
