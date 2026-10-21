"""Build the smoke report exclusively from saved metrics."""

import sys

from atc_fatigue.cli import main

if __name__ == "__main__":
    main(["report", *sys.argv[1:]])
