#!/usr/bin/env python3
"""Record actual CUDA/OpenGL frames, then encode GitHub-friendly GIF/WebP media.

VHS records terminal cells rather than an OpenGL window. Using the application's
fixed-step readback gives reproducible footage without desktop/cursor clutter.
Only Python's standard library and ffmpeg are required; gifsicle is optional.
"""
import argparse
import pathlib
import shlex
import shutil
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", default="", help="e.g. prime-run; empty for --staged")
    parser.add_argument("--staged", action="store_true", help="explicit CUDA -> CPU -> GL fallback")
    parser.add_argument("--seconds", type=int, default=24, choices=range(1, 121), metavar="1..120")
    parser.add_argument("--count", type=int, default=200000)
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("docs"))
    args = parser.parse_args()
    if not shutil.which("ffmpeg"):
        parser.error("ffmpeg is required")
    args.output.mkdir(parents=True, exist_ok=True)
    scratch = pathlib.Path("/tmp/opencode")
    temp_root = scratch if scratch.is_dir() else None
    with tempfile.TemporaryDirectory(prefix="particle-atlas-", dir=temp_root) as tmp:
        frames = pathlib.Path(tmp) / "frames"
        command = ([] if args.staged else shlex.split(args.runner)) + [
            "./build/particles", "--demo", "--count", str(args.count),
            "--width", "960", "--height", "540", "--frames", str(args.seconds * 60),
            "--capture", str(frames),
        ]
        if args.staged:
            command.append("--staged")
        subprocess.run(command, check=True)
        inputs = ["ffmpeg", "-hide_banner", "-loglevel", "warning", "-y",
                  "-framerate", "30", "-i", str(frames / "frame-%05d.ppm")]
        palette = pathlib.Path(tmp) / "palette.png"
        # A full tour becomes a tighter 12s montage: four mature seconds/scene.
        montage = "select='gte(mod(t,8),2)*lt(mod(t,8),6)',setpts=N/(30*TB)," if args.seconds >= 24 else ""
        filters = montage + "fps=15,scale=640:-1:flags=lanczos"
        subprocess.run(inputs + ["-vf", filters + ",palettegen=max_colors=128:stats_mode=diff",
                                "-frames:v", "1", "-update", "1", str(palette)], check=True)
        gif = args.output / "demo.gif"
        subprocess.run(inputs + ["-i", str(palette), "-lavfi",
                                f"{filters}[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=4",
                                "-loop", "0", str(gif)], check=True)
        subprocess.run(inputs + ["-vf", montage + "fps=20,scale=960:-1:flags=lanczos,format=yuv420p",
                                "-c:v", "libwebp_anim", "-quality", "65", "-loop", "0",
                                str(args.output / "demo.webp")], check=True)
        # A still is useful for social previews and quick visual regression checks.
        still = frames / f"frame-{min(90, args.seconds * 30 - 1):05d}.ppm"
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                        "-i", str(still), "-frames:v", "1", "-update", "1",
                        str(args.output / "preview.png")], check=True)
        if shutil.which("gifsicle"):
            subprocess.run(["gifsicle", "-O3", "-b", str(gif)], check=True)
    print(f"Recorded {args.seconds}s of real CUDA simulation ({'staged' if args.staged else 'interop'}).")
    for name in ("demo.gif", "demo.webp", "preview.png"):
        file = args.output / name
        print(f"  {file}: {file.stat().st_size / 1048576:.2f} MiB")


if __name__ == "__main__":
    main()
