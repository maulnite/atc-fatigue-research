"""Save a read-only runtime audit."""

import sys

from atc_fatigue.cli import main

if __name__ == "__main__":
    main(["environment", *sys.argv[1:]])
