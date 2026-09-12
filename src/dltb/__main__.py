"""`python -m dltb` is not a command -- the tools are console scripts."""

import sys

TOOLS = ("dltb-oneshot", "dltb-iterate", "dltb-continuous")

if __name__ == "__main__":
    print("dltb is a library; run one of its tools instead:", file=sys.stderr)
    for tool in TOOLS:
        print(f"  uv run {tool} --help", file=sys.stderr)
    sys.exit(2)
