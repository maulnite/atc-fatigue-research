"""Extract features from a previously prepared audio manifest."""

import sys

from atc_fatigue.cli import main

if __name__ == "__main__":
    main(["audio", *sys.argv[1:]])
