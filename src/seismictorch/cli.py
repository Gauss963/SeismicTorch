"""One command entry point; optional research dependencies load only when used."""

import argparse
import runpy
import sys

COMMANDS = {
    "train": "seismictorch.training.supervised",
    "qsis": "seismictorch.qsis.candidate",
    "predict": "seismictorch.inference.waveform",
}
TOOLS = {
    "review": "seismictorch.qsis.review",
    "split": "seismictorch.qsis.split",
    "reference": "seismictorch.qsis.reference",
    "array-review": "seismictorch.qsis.array_review",
    "consensus": "seismictorch.qsis.consensus",
    "public": "seismictorch.qsis.public",
    "public-score": "seismictorch.qsis.public_score",
    "catalog": "seismictorch.qsis.catalog",
    "noise-audit": "seismictorch.qsis.noise_audit",
    "array-proposals": "seismictorch.qsis.array_proposals",
    "redpan": "seismictorch.qsis.redpan",
    "pilot": "seismictorch.training.pilot",
    "transfer": "seismictorch.training.transfer",
    "cwa": "seismictorch.training.cwa",
    "utrans": "seismictorch.qsis.utrans",
}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description="SeismicTorch waveform training, prediction and QSIS tools.")
    parser.add_argument("command", choices=(*COMMANDS, "tools", "legacy-run"))
    if not argv or argv[0] in {"-h", "--help"}:
        parser.print_help()
        return
    command = parser.parse_args(argv[:1]).command
    remaining = argv[1:]
    if command == "legacy-run":
        from seismictorch.legacy import main as legacy_main
        legacy_main(remaining)
        return
    if command == "tools":
        tool_parser = argparse.ArgumentParser(prog="seismictorch tools", description="QSIS review, acquisition and experiments.")
        tool_parser.add_argument("tool", choices=tuple(TOOLS))
        if not remaining or remaining[0] in {"-h", "--help"}:
            tool_parser.print_help()
            return
        tool = tool_parser.parse_args(remaining[:1]).tool
        module = TOOLS[tool]
        program = f"seismictorch tools {tool}"
        remaining = remaining[1:]
    else:
        module = COMMANDS[command]
        program = f"seismictorch {command}"
    previous = sys.argv
    try:
        sys.argv = [program, *remaining]
        runpy.run_module(module, run_name="__main__")
    finally:
        sys.argv = previous
