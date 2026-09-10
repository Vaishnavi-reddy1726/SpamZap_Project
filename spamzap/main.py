"""
main.py
-------
Convenience entry point for SpamZap.

    python main.py train                     -> trains the model, prints metrics
    python main.py predict "some message"     -> classifies a single message
    python main.py predict -i                 -> interactive classification
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent / "src"))


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("train", "predict"):
        print(__doc__)
        sys.exit(1)

    command = sys.argv.pop(1)
    if command == "train":
        import train
        train.main()
    elif command == "predict":
        import predict
        predict.main()


if __name__ == "__main__":
    main()
