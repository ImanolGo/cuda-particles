#!/usr/bin/env python3
"""CLI validation and bounded GPU/GL smoke tests. Run from the repository root."""
import argparse
import pathlib
import shlex
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staged", action="store_true")
    parser.add_argument("--runner", default="")
    args = parser.parse_args()
    binary = "./build/particles"
    for options in (["--count", "nan"], ["--fade", "inf"], ["--size", "oops"],
                    ["--count", "0"], ["--count", "1.5"], ["--frames"],
                    ["--scene", "missing"], ["--capture", "/tmp/opencode/not-used"]):
        result = subprocess.run([binary] + options, capture_output=True, timeout=10)
        assert result.returncode != 0, options
    assert subprocess.run([binary, "--help"], capture_output=True).returncode == 0
    prefix = [] if args.staged else shlex.split(args.runner)
    with tempfile.TemporaryDirectory(prefix="particle-test-", dir="/tmp/opencode") as tmp:
        protected = pathlib.Path(tmp) / "protected"
        protected.mkdir()
        sentinel = protected / "keep.txt"
        sentinel.write_text("do not overwrite")
        result = subprocess.run([binary, "--frames", "2", "--capture", str(protected)],
                                capture_output=True, timeout=10)
        assert result.returncode != 0 and sentinel.read_text() == "do not overwrite"
        for scene in ("orbit", "aurora", "ink"):
            directory = pathlib.Path(tmp) / scene
            command = prefix + [binary, "--scene", scene, "--count", "1027", "--frames", "8",
                                "--width", "320", "--height", "240", "--capture", str(directory)]
            if args.staged:
                command.append("--staged")
            subprocess.run(command, check=True, timeout=60)
            images = sorted(directory.glob("*.ppm"))
            assert len(images) == 4
            for image in images:
                data = image.read_bytes()
                assert data.startswith(b"P6\n320 240\n255\n")
                assert len(data) == len(b"P6\n320 240\n255\n") + 320 * 240 * 3
            assert images[0].read_bytes() != images[-1].read_bytes(), "scene did not advance"
    print("CLI validation + all three CUDA/render/capture scenes passed.")


if __name__ == "__main__":
    main()
