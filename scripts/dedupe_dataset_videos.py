#!/usr/bin/env python
"""Share one copy of each camera stream between dataset variants that differ
only in their action or state columns.

One dataset cannot serve every option. LeRobot binds training to the single
column named `action`, and `dataset_to_policy_features` types *any* key starting
with `observation` or `action` as a policy feature -- which is how a stray
`action.joint_position` column became a second ACTION feature and broke policy
construction. So alternative encodings have to live in separate dataset
directories.

But they do not have to live in separate bytes. A variant changes only the
parquet: for the r75 corpus each RGB variant is 168 MB, of which 161 MB is the
same three camera streams re-encoded from the same clips in the same order, and
the encoder is deterministic enough that the files are byte-identical --
verified by checksum across five variants.

This replaces those duplicates with hard links. Hard links rather than symlinks
because they are invisible to every reader: same inode, no path resolution, and
nothing in LeRobot needs to know. The exports are immutable in practice (the
build job refuses to overwrite an existing dataset), so sharing an inode is
safe; a writer would have to break the link first.

Files are linked only when their content hashes are equal, never on the strength
of a matching name or size.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from collections import defaultdict
from pathlib import Path

#: Read in chunks: these files are ~50 MB each and there is no reason to hold
#: one in memory to hash it.
CHUNK = 1 << 20


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(CHUNK):
            sha.update(block)
    return sha.hexdigest()


def video_files(dataset: Path) -> list[Path]:
    return sorted(p for p in (dataset / "videos").rglob("*.mp4") if p.is_file())


def plan(datasets: list[Path]) -> dict[tuple[str, str], list[Path]]:
    """Group identical files by (path within the dataset, content hash).

    Keyed on the relative path as well as the hash so that a file is only ever
    shared with its own counterpart in another variant. Two different cameras
    that happened to encode identically would still be kept apart, which costs
    nothing and keeps each dataset's tree self-explanatory.
    """
    groups: dict[tuple[str, str], list[Path]] = defaultdict(list)
    for dataset in datasets:
        for path in video_files(dataset):
            groups[(str(path.relative_to(dataset)), digest(path))].append(path)
    return groups


def link(canonical: Path, duplicate: Path) -> int:
    """Point `duplicate` at `canonical`'s inode. Returns bytes reclaimed.

    Written through a temporary name and an atomic replace, so an interruption
    leaves either the original file or the link, never a missing one.
    """
    if canonical.stat().st_ino == duplicate.stat().st_ino:
        return 0
    size = duplicate.stat().st_size
    temporary = duplicate.with_suffix(duplicate.suffix + ".linking")
    os.link(canonical, temporary)
    os.replace(temporary, duplicate)
    return size


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="+", type=Path,
                        help="Dataset directories to share video between.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    datasets = [d for d in args.datasets if (d / "videos").is_dir()]
    if not datasets:
        raise SystemExit("None of the given directories has a videos/ subdirectory")
    print(f"scanning {len(datasets)} dataset(s) with video")

    reclaimed = shared = 0
    for (relative, _), paths in sorted(plan(datasets).items()):
        if len(paths) < 2:
            continue
        canonical, duplicates = paths[0], paths[1:]
        inodes = {p.stat().st_ino for p in paths}
        if len(inodes) == 1:
            continue  # already shared
        print(f"  {relative}: {len(paths)} copies -> 1")
        for duplicate in duplicates:
            if args.dry_run:
                reclaimed += duplicate.stat().st_size
            else:
                reclaimed += link(canonical, duplicate)
            shared += 1

    verb = "would reclaim" if args.dry_run else "reclaimed"
    print(f"\n{verb} {reclaimed / 1e6:.0f} MB by sharing {shared} file(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
