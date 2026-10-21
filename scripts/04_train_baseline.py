"""Train the fixed subject-independent MRL eye-state baseline."""

import sys

from atc_fatigue.cli import main

if __name__ == "__main__":
    main(["train-mrl", *sys.argv[1:]])
