"""Run the genuine paired RAVDESS emotion demo."""

import sys

from atc_fatigue.cli import main

if __name__ == "__main__":
    main(["fusion", *sys.argv[1:]])
