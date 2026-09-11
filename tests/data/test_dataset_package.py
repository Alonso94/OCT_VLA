import json

from oct_vla.data.package import build_dataset_manifest


def test_manifest_records_dataset_metadata_and_file_hashes(tmp_path):
    (tmp_path / "meta").mkdir()
    (tmp_path / "data").mkdir()
    info = {"codebase_version": "v3.0", "total_episodes": 2, "total_frames": 10, "features": {}}
    (tmp_path / "meta" / "info.json").write_text(json.dumps(info))
    (tmp_path / "data" / "frames.parquet").write_bytes(b"portable")

    manifest = build_dataset_manifest(tmp_path)

    assert manifest["episodes"] == 2
    assert [entry["path"] for entry in manifest["files"]] == [
        "data/frames.parquet",
        "meta/info.json",
    ]
    assert len(manifest["files"][0]["sha256"]) == 64
