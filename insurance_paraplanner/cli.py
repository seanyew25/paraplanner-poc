from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .graph import run_advisory


def load_profile(value: str) -> dict[str, Any]:
    path = Path(value)
    if path.is_file():
        profile = json.loads(path.read_text(encoding="utf-8"))
    else:
        profile = json.loads(value)
    if not isinstance(profile, dict):
        raise ValueError("Profile JSON must contain an object at the top level")
    return profile


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the insurance paraplanner")
    parser.add_argument(
        "profile",
        help="Path to a profile JSON file or an inline JSON object",
    )
    parser.add_argument("--database", type=Path, default=None)
    parser.add_argument("--chroma-dir", type=Path, default=None)
    args = parser.parse_args()

    profile = load_profile(args.profile)
    kwargs = {}
    if args.database is not None:
        kwargs["database_path"] = args.database
    if args.chroma_dir is not None:
        kwargs["chroma_path"] = args.chroma_dir

    result = run_advisory(profile, **kwargs)
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
