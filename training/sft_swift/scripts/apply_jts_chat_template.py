#!/usr/bin/env python3
"""Apply the Judge-then-Solve Qwen3 chat template to an HF checkpoint."""

import argparse
import json
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-dir",
        required=True,
        help="HuggingFace checkpoint directory whose tokenizer_config.json will be updated.",
    )
    parser.add_argument(
        "--template-config",
        default=None,
        help="Tokenizer config containing the Judge-then-Solve chat template.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create tokenizer_config.json.bak before writing.",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    template_config = (
        Path(args.template_config)
        if args.template_config
        else script_dir.parent / "qwen3_jts_tokenizer_config.json"
    )
    checkpoint_dir = Path(args.checkpoint_dir)
    target_config = checkpoint_dir / "tokenizer_config.json"

    if not template_config.is_file():
        raise FileNotFoundError(f"Template config not found: {template_config}")
    if not checkpoint_dir.is_dir():
        raise NotADirectoryError(f"Checkpoint directory not found: {checkpoint_dir}")

    template_data = json.loads(template_config.read_text())
    if "chat_template" not in template_data:
        raise KeyError(f"No chat_template found in {template_config}")

    if target_config.exists() and not args.no_backup:
        backup_config = checkpoint_dir / "tokenizer_config.json.bak"
        shutil.copy2(target_config, backup_config)

    target_config.write_text(json.dumps(template_data, ensure_ascii=False, indent=4) + "\n")
    print(f"Wrote Judge-then-Solve tokenizer config to {target_config}")


if __name__ == "__main__":
    main()
