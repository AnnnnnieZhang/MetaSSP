"""Split the Pix3D training set into a labeled and an unlabeled subset.

For every category, the first ``ceil(p% * N)`` samples of ``all.json`` (in file
order) form the labeled subset; the remaining samples form the unlabeled subset.
The two subsets are written next to ``all.json`` as ``<p>%.json`` and ``<100-p>%.json``.

Usage:
    python tools/split_pix3d.py --percent 10
"""
import os
import json
import math
import argparse


def count_categories(data):
    """Number of samples per category; each record is [image, object id, category]."""
    counts = {}
    for record in data:
        counts[record[2]] = counts.get(record[2], 0) + 1
    return counts


def split(data, percent):
    thresholds = {c: math.ceil(n * percent / 100.0) for c, n in count_categories(data).items()}
    labeled, unlabeled, taken = [], [], {}
    for record in data:
        category = record[2]
        if taken.get(category, 0) < thresholds[category]:
            labeled.append(record)
            taken[category] = taken.get(category, 0) + 1
        else:
            unlabeled.append(record)
    return labeled, unlabeled


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--split-file', default='dataset/Pix3D/split/train/all.json',
                        help='full training split (default: %(default)s)')
    parser.add_argument('--percent', type=int, default=10,
                        help='percentage of labeled samples per category (default: %(default)s)')
    args = parser.parse_args()
    if not 0 < args.percent < 100:
        parser.error('--percent must be an integer in (0, 100)')

    with open(args.split_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    labeled, unlabeled = split(data, args.percent)

    total = count_categories(data)
    kept = count_categories(labeled)
    print('labeled / total samples per category:')
    for category in total:
        print(f'  {category}: {kept.get(category, 0)}/{total[category]}')

    out_dir = os.path.dirname(args.split_file)
    for name, subset in ((f'{args.percent}%.json', labeled), (f'{100 - args.percent}%.json', unlabeled)):
        path = os.path.join(out_dir, name)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(subset, f, indent=4, ensure_ascii=False)
        print(f'saved {len(subset)} samples to {path}')


if __name__ == "__main__":
    main()
