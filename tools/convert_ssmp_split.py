"""Convert the Pix3D split file of SSMP (``Pix3D_10.json``) to the split format of this repo.

SSMP stores, per category, lists of image ids for ``train`` / ``unlabel`` / ``val`` / ``test``.
Each list is written to ``<output-dir>/<split>.json`` as records
``["img/<category>/<id>.jpg", "255", "<category>"]``. Only needed to train on the
same split as SSMP for comparison.

Usage:
    python tools/convert_ssmp_split.py --input dataset/Pix3D/SSMP/Pix3D_10.json
"""
import os
import json
import argparse

SPLITS = ("train", "unlabel", "val", "test")


def convert(input_path, output_dir):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    splits = {name: [] for name in SPLITS}
    for entry in data:
        category = entry.get("taxonomy_id", "").strip()
        for name in SPLITS:
            for file_id in entry.get(name, []):
                splits[name].append([f"img/{category}/{file_id}.jpg", "255", category])

    os.makedirs(output_dir, exist_ok=True)
    for name, records in splits.items():
        output_path = os.path.join(output_dir, f"{name}.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=4, ensure_ascii=False)
        print(f"saved {len(records)} records to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default="dataset/Pix3D/SSMP/Pix3D_10.json")
    parser.add_argument("--output-dir", default=None, help="default: directory of --input")
    args = parser.parse_args()
    convert(args.input, args.output_dir or os.path.dirname(args.input))
