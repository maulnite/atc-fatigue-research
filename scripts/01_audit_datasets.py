"""Check public source pages; file access and licenses still require review."""

import sys

from atc_fatigue.cli import main

if __name__ == "__main__":
    main(["audit", *sys.argv[1:]])
