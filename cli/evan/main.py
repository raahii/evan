import argparse
import sys

from .compute import do_compute_command
from .config import EVAN_CACHE_DIR
from .plot import do_plot_command


def main():
    if not EVAN_CACHE_DIR.exists():
        EVAN_CACHE_DIR.mkdir(parents=True)

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("command", choices=["compute"], help="command")
    args, rest_args = parser.parse_known_args(sys.argv[1:])

    if args.command == "compute":
        do_compute_command(rest_args)
    else:
        raise NotImplementedError

    # args = parser.parse_args()
    # if hasattr(args, "handler"):
    #     args.handler(args)
    # else:
    #     parser.print_help()
