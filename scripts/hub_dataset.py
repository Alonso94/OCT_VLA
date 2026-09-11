#!/usr/bin/env python
"""Publish or retrieve an immutable portable LeRobot dataset release."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _require_hub():
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ModuleNotFoundError as error:
        raise RuntimeError("Install huggingface_hub in the policy environment.") from error
    return HfApi, snapshot_download


def _read_info(root: Path) -> dict:
    path = root / "meta" / "info.json"
    if not path.is_file():
        raise FileNotFoundError(f"Not a finalized LeRobot dataset: {path} is missing")
    return json.loads(path.read_text())


def _card(repo_id: str, release: str, info: dict) -> str:
    return "\n".join(
        (
            "---",
            "tags:",
            "- lerobot",
            "- robotics",
            "- oct-vla",
            "---",
            "",
            f"# {repo_id}",
            "",
            f"Portable OCT-VLA LeRobot v3 release `{release}`.",
            "",
            f"* Episodes: {info['total_episodes']}",
            f"* Frames: {info['total_frames']}",
            f"* FPS: {info['fps']}",
            "",
            "`dataset_manifest.json` contains SHA-256 checksums for every file.",
            "",
        )
    )


def publish(args: argparse.Namespace) -> int:
    from oct_vla.data.package import build_dataset_manifest

    root = args.dataset.resolve()
    info = _read_info(root)
    manifest = build_dataset_manifest(root)
    HfApi, _ = _require_hub()
    api = HfApi(token=args.token)
    api.create_repo(args.repo_id, repo_type="dataset", private=args.private, exist_ok=True)
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="dataset",
        folder_path=root,
        commit_message=f"Publish OCT-VLA dataset {args.release}",
    )
    with tempfile.TemporaryDirectory() as directory:
        temp = Path(directory)
        (temp / "README.md").write_text(_card(args.repo_id, args.release, info))
        (temp / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        commit = api.upload_folder(
            repo_id=args.repo_id,
            repo_type="dataset",
            folder_path=temp,
            commit_message=f"Add manifest for {args.release}",
        )
    api.create_tag(
        repo_id=args.repo_id,
        repo_type="dataset",
        tag=args.release,
        revision=commit.oid,
        exist_ok=True,
    )
    print(f"published https://huggingface.co/datasets/{args.repo_id}/tree/{args.release}")
    return 0


def download(args: argparse.Namespace) -> int:
    _, snapshot_download = _require_hub()
    destination = args.output.resolve()
    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=args.release,
        local_dir=destination,
        token=args.token,
    )
    _read_info(destination)
    print(f"downloaded {args.repo_id}@{args.release} to {destination}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    publish_parser = commands.add_parser("publish")
    publish_parser.add_argument("dataset", type=Path)
    publish_parser.add_argument("repo_id", help="Hub dataset repo, e.g. org/oct-vla-rgb")
    publish_parser.add_argument("--release", required=True)
    publish_parser.add_argument("--private", action="store_true")
    publish_parser.add_argument(
        "--token", help="Hub write token; otherwise use the logged-in token"
    )
    publish_parser.set_defaults(func=publish)
    download_parser = commands.add_parser("download")
    download_parser.add_argument("repo_id")
    download_parser.add_argument("--release", required=True)
    download_parser.add_argument("--output", required=True, type=Path)
    download_parser.add_argument("--token", help="Hub read token for private datasets")
    download_parser.set_defaults(func=download)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
