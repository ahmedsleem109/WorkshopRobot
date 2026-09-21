"""T12: cut the ~90 s montage from the rendered clips in media/.

    python scripts/make_montage.py [--out media/montage.mp4] [--ffmpeg PATH]

Each entry in SEGMENTS is one clip, optionally sped up, with a caption burnt in; TITLES are
full-frame cards. Everything is normalised to 960x540 @ 25 fps so the concat demuxer can join
them without re-encoding twice. Re-run it after new clips are rendered -- it is cheap.
"""
import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
W, H, FPS = 960, 540, 25
BG = "0x0E1116"
FONT = r"C\:/Windows/Fonts/arialbd.ttf"          # ffmpeg filter escaping, not a path typo

# (kind, ...) -- "card": (title, subtitle, seconds); "clip": (file, speed, caption, seconds|None)
TIMELINE = [
    ("card", "Bring me the 10 mm wrench", "A quadruped with an arm, in simulation", 3.0),

    ("card", "1  Grasping from the rack", "5 tools, 125/125 held  -  on legs, no fixed base", 2.5),
    ("clip", "grasp_wrench_10mm.mp4", 1.6, "10 mm wrench", None),
    ("clip", "grasp_pliers.mp4", 1.6, "pliers", None),
    ("clip", "grasp_tape_roll.mp4", 1.6, "tape roll  -  grasped radially at the crown", None),

    # One spoken command, all the way through: locate, grasp, cross the 12 cm step, hand over.
    # Every clip below is a trial the suite SCORED as a success, rendered from its own seed.
    ("card", "2  One command, end to end", "locate, grasp, cross the 12 cm step, hand over", 2.5),
    ("clip", "s8_nominal_wrench10.mp4", 5.0, "\"bring me the 10mm wrench\"", None, "top"),
    ("clip", "s8_nominal_screwdriver.mp4", 5.0, "\"bring me the screwdriver\"", None, "top"),
    ("clip", "s8_nominal_tape.mp4", 5.0, "\"I need the roll of tape\"", None, "top"),

    ("card", "3  Table to table", "pick, walk, place inside the named zone", 2.5),
    ("clip", "s8_transfer.mp4", 4.0, "to the far table, 3 mm from the zone centre  -  9/10", None, "top"),

    ("card", "4  When it goes wrong", "a dropped tool, a blocked route, an ambiguous name, "
     "a missing tool, a change of mind", 3.0),
    ("clip", "s8_recover_drop.mp4", 7.0, "dropped mid-carry, asks for it back  -  9/10", None, "top"),
    ("clip", "s8_recover_obstacle.mp4", 7.0, "a box lands on the route, detours and places  -  10/10", None, "top"),
    ("clip", "s8_recover_ambiguous.mp4", 5.0, "\"bring me the wrench\"  ->  \"which one?\"  -  8/10", None, "top"),
    ("clip", "s8_recover_missing.mp4", 1.0, "not in the rack: says so, does not guess  -  10/10", None, "top"),
    ("clip", "s8_recover_retarget.mp4", 7.0, "\"actually, the 13 mm\", mid-task  -  6/10", None, "top"),

    ("card", "5  Learning the grasp", "SmolVLA on 1,128 demonstrations  -  0/20 to 18/20 on scenes "
     "it never saw", 3.0),

    ("card", "Grasp 125/125   Place 120/125 and 124/125   End to end 60/70",
     "Grounding 10.7 mm median   VLA grasp 18/20 held out   one 6 GB GPU, no API keys", 4.0),
]


def esc(t: str) -> str:
    """Escape text for ffmpeg's drawtext."""
    return (t.replace("\\", "\\\\").replace(":", r"\:")
            .replace("'", "’").replace("%", r"\%"))


# Arial Bold advances about 0.56 em per character averaged over this deck's text. Good enough to
# decide line breaks; the card is centred, so a few px of error is invisible.
CHAR_EM = 0.56
MARGIN = 60


def wrap(text, fontsize, max_lines):
    """Greedy wrap, shrinking the font until the text fits max_lines within the frame."""
    # The deck separates statistics on one line by a RUN of spaces ("Grasp 125/125   Place ...").
    # Wrapping splits on whitespace, which would collapse those to single spaces and leave the
    # line reading as one run-on sentence, so promote them to a visible separator first.
    text = re.sub(r"\s{2,}(?:[-–—]\s{2,})?", " · ", text.strip())
    while True:
        limit = max(1, int((W - 2 * MARGIN) / (fontsize * CHAR_EM)))
        lines, cur = [], ""
        for word in text.split():
            trial = f"{cur} {word}".strip()
            if len(trial) <= limit or not cur:
                cur = trial
            else:
                lines.append(cur)
                cur = word
        if cur:
            lines.append(cur)
        if len(lines) <= max_lines or fontsize <= 14:
            return lines, fontsize
        fontsize -= 2


def _text(line, fontsize, colour, y):
    # expansion=none: drawtext otherwise runs its own strftime-style pass over the text and a
    # bare '%' makes it drop the WHOLE string -- which is how the closing card lost its subtitle
    # ("2.8% miss") while only warning "Stray %" to stderr.
    return (f"drawtext=fontfile='{FONT}':text='{esc(line)}':fontcolor={colour}:"
            f"fontsize={fontsize}:expansion=none:x=(w-text_w)/2:y={y}")


def card(ff, title, subtitle, secs, out):
    tl, tsz = wrap(title, 44, 2)
    sl, ssz = wrap(subtitle, 26, 2)
    parts = []
    # the title block sits above the midline, the subtitle below it, whatever the line counts
    y = H // 2 - 40 - len(tl) * (tsz + 10)
    for line in tl:
        parts.append(_text(line, tsz, "white", y))
        y += tsz + 10
    y = H // 2 + 10
    for line in sl:
        parts.append(_text(line, ssz, "0x9FB3C8", y))
        y += ssz + 8
    run(ff, ["-f", "lavfi", "-i", f"color=c={BG}:s={W}x{H}:r={FPS}:d={secs}", "-vf",
             ",".join(parts)], out)


def clip(ff, src, speed, caption, secs, out, pos="bottom"):
    """pos="top" for clips that already burn in their own caption bar.

    make_orch_video.py draws the command, the orchestrator state and the robot/human dialogue
    across the bottom 118 px of every frame it renders. Stamping the montage's own caption at
    h-64 lands squarely on top of that -- the suite score overlapped "state: ASK_HUMAN" and the
    dialogue line, and both became unreadable. The top-left is free on those clips (the wrist
    camera inset sits top-RIGHT), so the caption goes there instead.
    """
    vf = [f"setpts=PTS/{speed}", f"scale={W}:{H}:force_original_aspect_ratio=decrease",
          f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color={BG}", f"fps={FPS}"]
    if caption:
        box_y, text_y = ("0", "18") if pos == "top" else ("h-64", "h-46")
        vf.append(f"drawbox=x=0:y={box_y}:w=iw:h=64:color=black@0.55:t=fill")
        vf.append(f"drawtext=fontfile='{FONT}':text='{esc(caption)}':fontcolor=white:"
                  f"fontsize=26:expansion=none:x=28:y={text_y}")
    args = ["-i", str(src), "-vf", ",".join(vf)]
    if secs:
        args += ["-t", str(secs)]
    run(ff, args, out)


def run(ff, args, out):
    cmd = [ff, "-y", "-v", "error"] + args + ["-an", "-c:v", "libx264", "-preset", "medium",
                                              "-crf", "20", "-pix_fmt", "yuv420p", str(out)]
    subprocess.run(cmd, check=True)


def duration(ff, path):
    probe = str(Path(ff).with_name(Path(ff).name.replace("ffmpeg", "ffprobe")))
    p = subprocess.run([probe, "-v", "error", "-show_entries",
                        "format=duration", "-of", "csv=p=0", str(path)],
                       capture_output=True, text=True, check=True)
    return float(p.stdout.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "media" / "montage.mp4"))
    ap.add_argument("--ffmpeg", default=shutil.which("ffmpeg") or
                    "C:/ffmpeg/ffmpeg-2023-04-06-git-b564ad8eac-full_build/bin/ffmpeg.exe")
    ap.add_argument("--media", default=str(ROOT / "media"))
    args = ap.parse_args()
    ff, media = args.ffmpeg, Path(args.media)
    if not Path(ff).exists():
        sys.exit(f"ffmpeg not found: {ff}  (pass --ffmpeg)")

    work = ROOT / "runs" / "montage_parts"
    work.mkdir(parents=True, exist_ok=True)
    parts = []
    for i, entry in enumerate(TIMELINE):
        out = work / f"{i:02d}.mp4"
        if entry[0] == "card":
            card(ff, entry[1], entry[2], entry[3], out)
        else:
            src = media / entry[1]
            if not src.exists():
                print(f"  skip (missing): {src.name}")
                continue
            # a 6th element overrides caption placement; default keeps the old behaviour
            clip(ff, src, entry[2], entry[3], entry[4], out,
                 entry[5] if len(entry) > 5 else "bottom")
        parts.append(out)
        print(f"  {out.name}  {duration(ff, out):5.1f}s  {entry[1][:48]}")

    lst = work / "parts.txt"
    lst.write_text("\n".join(f"file '{p.as_posix()}'" for p in parts))
    subprocess.run([ff, "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
                    "-c", "copy", args.out], check=True)
    print(f"\n{args.out}  {duration(ff, args.out):.1f}s  "
          f"{Path(args.out).stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
