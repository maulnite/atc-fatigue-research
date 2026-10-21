"""Process DROZY landmarks with original timestamps."""

import sys

from atc_fatigue.cli import main

if __name__ == "__main__":
    main(["drozy", *sys.argv[1:]])
