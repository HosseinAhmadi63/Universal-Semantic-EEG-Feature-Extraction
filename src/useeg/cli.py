from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from typing import Any

from .analysis import run_analysis
from .config import (
    config_hash,
    dataset_map,
    load_config,
    resolve_path,
    selected_dataset_names,
    with_selection,
)
from .data import cache_dataset, download_dataset
from .experiment import (
    classify_dataset,
    extract_dataset_features,
    run_ablation,
    train_dataset_autoencoder,
)
from .plotting import make_figures
from .utils import set_global_seed
from .verify import verify_repository

STAGES = (
    "download",
    "preprocess",
    "train-autoencoder",
    "extract",
    "classify",
    "analyze",
    "figures",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="useeg")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in (*STAGES, "run", "verify"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--config", default="configs/paper.yaml")
        if command != "verify":
            subparser.add_argument("--datasets", nargs="+", default=["all"])
            subparser.add_argument("--subjects", nargs="+", type=int)
            subparser.add_argument("--force", action="store_true")
            subparser.add_argument("--verbose", action="store_true")
    return parser


def _configuration(arguments: argparse.Namespace) -> tuple[dict[str, Any], list[str]]:
    config = load_config(arguments.config)
    config = with_selection(config, arguments.datasets, arguments.subjects)
    names = selected_dataset_names(config, arguments.datasets)
    return config, names


def _announce(arguments: argparse.Namespace, message: str) -> None:
    if getattr(arguments, "verbose", False):
        print(message, flush=True)


def _download(config: dict[str, Any], names: list[str], arguments: argparse.Namespace) -> None:
    specifications = dataset_map(config)
    raw = resolve_path(config, "data_raw")
    for name in names:
        _announce(arguments, f"download {name}")
        download_dataset(
            specifications[name],
            raw,
            subjects=arguments.subjects,
            force=arguments.force,
            accept=bool(config["data"]["accept_dataset_terms"]),
        )


def _preprocess(config: dict[str, Any], names: list[str], arguments: argparse.Namespace) -> None:
    specifications = dataset_map(config)
    raw = resolve_path(config, "data_raw")
    processed = resolve_path(config, "data_processed")
    key = config_hash(config)
    for name in names:
        _announce(arguments, f"preprocess {name}")
        cache_dataset(
            specifications[name],
            raw,
            processed,
            subjects=arguments.subjects,
            preprocessing=config["preprocessing"],
            seed=int(config["seed"]),
            force=arguments.force,
            run_key=key,
        )


def _train(config: dict[str, Any], names: list[str], arguments: argparse.Namespace) -> None:
    for name in names:
        _announce(arguments, f"train-autoencoder {name}")
        train_dataset_autoencoder(config, name, force=arguments.force)


def _extract(config: dict[str, Any], names: list[str], arguments: argparse.Namespace) -> None:
    for name in names:
        _announce(arguments, f"extract {name}")
        extract_dataset_features(config, name, force=arguments.force)


def _classify(config: dict[str, Any], names: list[str], arguments: argparse.Namespace) -> None:
    for name in names:
        _announce(arguments, f"classify {name}")
        classify_dataset(config, name, force=arguments.force)


def _analyze(config: dict[str, Any], names: list[str], arguments: argparse.Namespace) -> None:
    ablation_dataset = str(config["analysis"]["ablation"]["dataset"])
    if ablation_dataset in names:
        _announce(arguments, f"ablation {ablation_dataset}")
        run_ablation(config, force=arguments.force)
    _announce(arguments, "analysis")
    run_analysis(config, names, force=arguments.force)


def _figures(config: dict[str, Any], names: list[str], arguments: argparse.Namespace) -> None:
    del names
    _announce(arguments, "figures")
    make_figures(config)


COMMANDS: dict[str, Callable[[dict[str, Any], list[str], argparse.Namespace], None]] = {
    "download": _download,
    "preprocess": _preprocess,
    "train-autoencoder": _train,
    "extract": _extract,
    "classify": _classify,
    "analyze": _analyze,
    "figures": _figures,
}


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "verify":
        config = load_config(arguments.config)
        report = verify_repository(config)
        print(f"verified {report['protocol']['profile']} profile")
        return 0
    config, names = _configuration(arguments)
    set_global_seed(int(config["seed"]), bool(config["deterministic"]))
    commands = STAGES if arguments.command == "run" else (arguments.command,)
    for command in commands:
        COMMANDS[command](config, names, arguments)
    print(f"completed {arguments.command} for {', '.join(names)} with run key {config_hash(config)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
