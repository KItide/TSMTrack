"""Export VTUAV-ST RGB and IR GT text files to one flat directory."""

import argparse
import filecmp
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = Path(
    "/home/dfc/Documents/dwh_documents/HMFT/HMFT-master/"
    "dataset/test_data/test_ST"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "RGBT_workspace/ground_truth/VTUAVST"
EXPECTED_SEQUENCES = 176


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE,
                        help="VTUAV-ST directory containing sequence folders")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Directory for flattened GT text files")
    args = parser.parse_args()

    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not source.is_dir():
        parser.error(f"source directory does not exist: {source}")

    sequences = sorted(path for path in source.iterdir() if path.is_dir())
    if len(sequences) != EXPECTED_SEQUENCES:
        parser.error(
            f"expected {EXPECTED_SEQUENCES} VTUAV-ST sequences, "
            f"found {len(sequences)} in {source}"
        )

    copies = []
    for sequence in sequences:
        for modality in ("rgb", "ir"):
            src = sequence / f"{modality}.txt"
            if not src.is_file() or src.stat().st_size == 0:
                parser.error(f"missing or empty annotation: {src}")
            dst = output / f"{sequence.name}_{modality}.txt"
            if dst.exists() and not filecmp.cmp(src, dst, shallow=False):
                parser.error(f"output already exists with different content: {dst}")
            copies.append((src, dst))

    output.mkdir(parents=True, exist_ok=True)
    copied = 0
    for src, dst in copies:
        if not dst.exists():
            shutil.copy2(src, dst)
            copied += 1

    print(f"VTUAV-ST: {len(sequences)} sequences, {len(copies)} GT TXT files")
    print(f"Saved to: {output} ({copied} new files)")


if __name__ == "__main__":
    main()
