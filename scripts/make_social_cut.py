"""Cut the two short social videos from the rendered clips in media/.

    python scripts/make_social_cut.py step        -> media/social_step.mp4
    python scripts/make_social_cut.py recovery    -> media/social_recovery.mp4
    python scripts/make_social_cut.py both

Square 1080x1080 for a feed, not 16:9: the source clip sits in a 1080x608 band in the middle and
the dark space above and below carries the captions, so nothing is cropped and the text is never
over the robot.

Every segment fades in and out of black, which is the transition. Beats that need to land -- a
fall, a recovery -- are slowed with `speed` < 1. Failure footage is REAL: the seeds in FAILS are
trials runs/eval/s7b scored against us.

A caption line is split on "|". Cards take (title, subtitle, seconds).
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
W = H = 1080
VID_H = 608                      # 1080 * 9/16, the source aspect, centred
VID_Y = (H - VID_H) // 2
FPS = 30
BG = "0x0B0E13"
ACCENT = "0x6FD3FF"
WARN = "0xFF6B6B"
FONT = "C\\:/Windows/Fonts/arialbd.ttf"
FADE = 0.25


def esc(t):
    return (t.replace("\\", "\\\\").replace(":", r"\:")
            .replace("'", "\u2019").replace("%", r"\%"))


def text(line, size, colour, y):
    return (f"drawtext=fontfile='{FONT}':text='{esc(line)}':fontcolor={colour}:fontsize={size}:"
            f"expansion=none:x=(w-text_w)/2:y={y}")


def stack(spec, size, colour, y0, gap):
    """spec is one string; "|" starts a new line."""
    if not spec:
        return []
    return [text(ln, size, colour, y0 + i * (size + gap))
            for i, ln in enumerate(spec.split("|"))]


def card(ff, title, subtitle, secs, out):
    vf = stack(title, 62, "white", 430, 16) + stack(subtitle, 34, ACCENT, 640, 12)
    vf.append(f"fade=t=in:st=0:d={FADE}")
    vf.append(f"fade=t=out:st={max(0.0, secs - FADE)}:d={FADE}")
    subprocess.run([ff, "-y", "-v", "error", "-f", "lavfi",
                    "-i", f"color=c={BG}:s={W}x{H}:r={FPS}:d={secs}",
                    "-vf", ",".join(vf), "-c:v", "libx264", "-preset", "veryfast",
                    "-crf", "20", "-pix_fmt", "yuv420p", "-an", str(out)], check=True)


def clip(ff, src, start, secs, speed, top, bottom, colour, out):
    """secs is OUTPUT duration; `start` seeks in the source before the speed change."""
    src_len = secs * speed
    vf = [f"setpts=PTS/{speed}",
          f"scale={W}:{VID_H}:force_original_aspect_ratio=decrease",
          f"pad={W}:{H}:(ow-iw)/2:{VID_Y}:color={BG}",
          f"fps={FPS}"]
    vf += stack(top, 52, colour, 96, 14)
    vf += stack(bottom, 34, "0x9FB3C8", H - 150, 10)
    vf.append(f"fade=t=in:st=0:d={FADE}")
    vf.append(f"fade=t=out:st={max(0.0, secs - FADE)}:d={FADE}")
    subprocess.run([ff, "-y", "-v", "error", "-ss", str(start), "-t", str(src_len),
                    "-i", str(src), "-vf", ",".join(vf), "-t", str(secs),
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                    "-pix_fmt", "yuv420p", "-an", str(out)], check=True)


# (kind, ...) -- "card": (title, subtitle, secs); "clip": (file, start, secs, speed, top, bottom, colour)
CUTS = {
    # Beat times were found by contact-sheeting each clip, not guessed: the fall in fail_step_a is
    # at 29-31 s, the clean descent in the nominal run at 27-33 s, the drop at ~34 s and the
    # ASK_HUMAN caption at ~40 s.
    "step": [
        ("clip", "fail_step_a.mp4", 28.5, 4.0, 0.45, "Watch what happens|at the step.", "12 cm down. Tool in the jaws.", WARN),
        ("card", "It fell 19 times|out of 20.", "Not a bug. A 12 cm step down.", 2.6),
        ("clip", "fail_step_b.mp4", 28.8, 3.2, 0.7, "Again.", "Different seed. Same step.", WARN),
        ("card", "8 of my 10|end-to-end failures", "were this one moment.", 2.6),
        ("card", "So I stopped guessing|and swept it.", "Falls per 20 trials, by step height", 2.6),
        ("card", "6cm 0   8cm 3   10cm 14|11cm 18   12cm 19   13cm 20", "The cliff is at 10 cm.", 3.2),
        ("card", "Then I retrained it.", "Step-height curriculum · 5.9M steps · JAX / MJX", 2.6),
        ("clip", "s8_nominal_wrench10.mp4", 27.0, 6.5, 1.0, "0 falls in 20.", "Same step. Same tool. Same seeds.", ACCENT),
        ("card", "Then I broke it|again.", "The fixed policy couldn’t turn around.", 2.8),
        ("card", "60/70  →  28/70", "So I trained the fix into a policy that CAN turn.|It lost the step. I reverted it the same night.", 3.6),
        ("card", "Disturbance rejection|and step descent", "are separable capabilities.|That is the actual finding.", 3.4),
        ("card", "github.com/ahmedsleem109|/WorkshopRobot", "MuJoCo · JAX · VLM grounding · SmolVLA", 3.2),
    ],
    "recovery": [
        ("clip", "s8_recover_drop.mp4", 34.0, 4.0, 0.5, "It just dropped|the wrench.", "Mid-carry. Nobody told it.", WARN),
        ("card", "No human in the loop.|No reset button.", "So what does it do?", 2.6),
        ("clip", "s8_recover_drop.mp4", 39.5, 5.5, 1.6, "It asks.", "Read the bottom line.   drop recovery 9/10", ACCENT),
        ("clip", "s8_recover_drop.mp4", 95.0, 4.5, 2.0, "Then finishes the job.", "Back to the rack, re-grasp, deliver.", ACCENT),
        ("card", "Four more ways|to go wrong.", "Each one is a scored suite. Not a demo.", 2.6),
        ("clip", "s8_recover_obstacle.mp4", 40.0, 6.0, 3.5, "A box lands on its route.", "Detects the stall, plans around it, places it.  10/10", ACCENT),
        ("clip", "s8_recover_ambiguous.mp4", 2.0, 5.0, 1.2, "“Bring me the wrench.”", "10mm or 13mm? It asks instead of guessing.  8/10", ACCENT),
        ("clip", "s8_recover_missing.mp4", 0.0, 4.0, 1.0, "The tool isn’t there.", "It says so. It does not grab the nearest thing.  10/10", ACCENT),
        ("clip", "fail_place.mp4", 33.5, 4.0, 1.0, "And sometimes|it just misses.", "1 transfer in 10 lands outside the zone.", WARN),
        ("card", "86% end to end.", "60 of 70 trials · 7 scenarios · fixed seeds.|All 10 failures counted and named.", 3.4),
        ("card", "One 6 GB laptop GPU.|No cloud. No API keys.", "MuJoCo · JAX · Qwen3-VL · SmolVLA", 3.0),
        ("card", "github.com/ahmedsleem109|/WorkshopRobot", "Full write-up. Failures included.", 3.0),
    ],
}


def build(name, ff, media, work, out):
    work.mkdir(parents=True, exist_ok=True)
    parts = []
    for i, e in enumerate(CUTS[name]):
        p = work / f"{name}_{i:02d}.mp4"
        if e[0] == "card":
            card(ff, e[1], e[2], e[3], p)
        else:
            src = media / e[1]
            if not src.exists():
                sys.exit(f"missing clip: {src}")
            clip(ff, src, e[2], e[3], e[4], e[5], e[6], e[7], p)
        parts.append(p)
        # the console here is cp1252; card titles carry arrows and curly quotes
        print(f"  {p.name}  {e[1][:44]}".encode("ascii", "replace").decode())
    lst = work / f"{name}.txt"
    lst.write_text("\n".join(f"file '{p.as_posix()}'" for p in parts))
    subprocess.run([ff, "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
                    "-c", "copy", str(out)], check=True)
    # only the FILENAME -- the ffmpeg build directory is also called "ffmpeg-<date>", and a
    # blanket replace turns C:/ffmpeg/ffmpeg-2023-.../bin/ffmpeg.exe into a path that does not exist
    probe = str(Path(ff).with_name(Path(ff).name.replace("ffmpeg", "ffprobe")))
    d = subprocess.run([probe, "-v", "error", "-show_entries",
                        "format=duration", "-of", "default=nw=1:nk=1", str(out)],
                       capture_output=True, text=True).stdout.strip()
    print(f"\n{out}  {float(d):.1f}s  {out.stat().st_size / 1e6:.1f} MB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["step", "recovery", "both"])
    ap.add_argument("--ffmpeg",
                    default="C:/ffmpeg/ffmpeg-2023-04-06-git-b564ad8eac-full_build/bin/ffmpeg.exe")
    ap.add_argument("--media", default=str(ROOT / "media"))
    args = ap.parse_args()
    media = Path(args.media)
    work = ROOT / "media" / "_social"
    names = ["step", "recovery"] if args.which == "both" else [args.which]
    for n in names:
        build(n, args.ffmpeg, media, work, media / f"social_{n}.mp4")


if __name__ == "__main__":
    main()
