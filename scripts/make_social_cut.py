"""Cut the two short social videos from the rendered clips in media/.

    python scripts/make_social_cut.py jobs       -> media/social_jobs.mp4      (3 complete runs)
    python scripts/make_social_cut.py recovery   -> media/social_recovery.mp4  (2 complete runs)
    python scripts/make_social_cut.py both

DESIGN, after a first attempt that did not work:

* A CASE is one complete task, start to finish, from a single rendered run. The viewer sees the
  typed command, the whole execution, and the outcome. No case is cut short and no two sources
  are intercut.
* Emphasis comes from SPEED RAMPS inside the same continuous shot -- fast through the walking,
  near real time through the moment that decides the trial -- not from cutting away. The parts of
  a case concatenate with no fade between them, so it reads as one take.
* Text is drawn ON the footage and only at the two moments it helps: the command for the first
  few seconds, the outcome for the last few. There are NO full-screen cards between cases. The
  source clips already burn in the command, the orchestrator state and the robot/human dialogue.
* Fades separate CASES, nothing else.

Square 1080x1080 for a feed: the 16:9 source sits in a band in the middle, so nothing is cropped.
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
OK = "0x7BE8A3"
WARN = "0xFF6B6B"
FONT = "C\\:/Windows/Fonts/arialbd.ttf"
FADE = 0.35
LEAD = 3.4                       # seconds the command stays up
TAIL = 3.0                       # seconds the outcome stays up


def esc(t):
    return (t.replace("\\", "\\\\").replace(":", r"\:")
            .replace("'", "\u2019").replace("%", r"\%"))


def lines(spec, size, colour, y0, gap, enable):
    out = []
    for i, ln in enumerate(spec.split("|")):
        out.append(f"drawtext=fontfile='{FONT}':text='{esc(ln)}':fontcolor={colour}:"
                   f"fontsize={size}:expansion=none:x=(w-text_w)/2:y={y0 + i * (size + gap)}:"
                   f"enable='{enable}'")
    return out


def part(ff, src, start, secs, speed, out):
    """One speed-ramped piece of a case. No text, no fade -- these concatenate seamlessly."""
    subprocess.run([ff, "-y", "-v", "error", "-ss", str(start), "-t", str(secs * speed),
                    "-i", str(src),
                    "-vf", (f"setpts=PTS/{speed},"
                            f"scale={W}:{VID_H}:force_original_aspect_ratio=decrease,"
                            f"pad={W}:{H}:(ow-iw)/2:{VID_Y}:color={BG},fps={FPS}"),
                    "-t", str(secs), "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                    "-pix_fmt", "yuv420p", "-an", str(out)], check=True)


def label_case(ff, raw, command, outcome, colour, secs, out):
    vf = lines(command, 50, "white", 100, 14, f"lt(t,{LEAD})")
    vf += lines(outcome, 46, colour, H - 195, 12, f"gt(t,{secs - TAIL})")
    vf.append(f"fade=t=in:st=0:d={FADE}")
    vf.append(f"fade=t=out:st={max(0.0, secs - FADE)}:d={FADE}")
    subprocess.run([ff, "-y", "-v", "error", "-i", str(raw), "-vf", ",".join(vf),
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                    "-pix_fmt", "yuv420p", "-an", str(out)], check=True)


# Each case: source, the typed command, the outcome line, its colour, and the speed ramp.
# (source_start, output_seconds, speed) -- 1.0 is real time, <1 is slow motion.
CASES = {
    "jobs": [
        dict(src="s8_nominal_wrench10.mp4", cmd='"bring me the 10mm wrench"',
             outcome="delivered", colour=OK,
             parts=[(0, 5.0, 4.0), (20, 2.4, 3.0), (27, 5.5, 1.3), (34, 7.2, 5.0)]),
        dict(src="fail_step_a.mp4", cmd='"bring me the 13mm wrench"',
             outcome="fell on the 12 cm step|8 of my 10 failures are this", colour=WARN,
             parts=[(0, 5.0, 4.0), (20, 2.7, 3.0), (28, 6.0, 0.8)]),
        dict(src="s8_recover_drop.mp4", cmd='"bring me the 13mm wrench"',
             outcome="dropped it, asked for it back, delivered", colour=OK,
             parts=[(0, 5.0, 6.0), (30, 5.0, 2.0), (40, 7.9, 7.0), (95, 4.0, 2.5)]),
    ],
    "recovery": [
        dict(src="s8_recover_obstacle.mp4", cmd='"put the pliers on the side table"',
             outcome="route blocked, detoured, placed", colour=OK,
             parts=[(0, 4.0, 5.0), (20, 12.0, 3.0), (60, 7.7, 6.0)]),
        dict(src="s8_recover_ambiguous.mp4", cmd='"bring me the wrench"',
             outcome="asked which one, then delivered", colour=OK,
             parts=[(0, 5.0, 2.0), (10, 10.4, 5.0)]),
    ],
}


def concat(ff, parts, lst, out, copy=True):
    lst.write_text("\n".join(f"file '{p.as_posix()}'" for p in parts))
    args = [ff, "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(lst)]
    args += ["-c", "copy"] if copy else ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                                         "-pix_fmt", "yuv420p"]
    subprocess.run(args + [str(out)], check=True)


def build(name, ff, media, work, out):
    work.mkdir(parents=True, exist_ok=True)
    cases = []
    for ci, case in enumerate(CASES[name]):
        src = media / case["src"]
        if not src.exists():
            sys.exit(f"missing clip: {src}")
        pieces = []
        for pi, (start, secs, speed) in enumerate(case["parts"]):
            p = work / f"{name}_{ci}_{pi}.mp4"
            part(ff, src, start, secs, speed, p)
            pieces.append(p)
        raw = work / f"{name}_{ci}_raw.mp4"
        concat(ff, pieces, work / f"{name}_{ci}.txt", raw)
        total = sum(s for _, s, _ in case["parts"])
        done = work / f"{name}_{ci}_done.mp4"
        label_case(ff, raw, case["cmd"], case["outcome"], case["colour"], total, done)
        cases.append(done)
        print(f"  case {ci}  {total:4.1f}s  {case['src']}")
    concat(ff, cases, work / f"{name}.txt", out, copy=False)
    probe = str(Path(ff).with_name(Path(ff).name.replace("ffmpeg", "ffprobe")))
    d = subprocess.run([probe, "-v", "error", "-show_entries", "format=duration",
                        "-of", "default=nw=1:nk=1", str(out)],
                       capture_output=True, text=True).stdout.strip()
    print(f"\n{out}  {float(d):.1f}s  {out.stat().st_size / 1e6:.1f} MB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["jobs", "recovery", "both"])
    ap.add_argument("--ffmpeg",
                    default="C:/ffmpeg/ffmpeg-2023-04-06-git-b564ad8eac-full_build/bin/ffmpeg.exe")
    ap.add_argument("--media", default=str(ROOT / "media"))
    args = ap.parse_args()
    media = Path(args.media)
    work = ROOT / "media" / "_social"
    names = ["jobs", "recovery"] if args.which == "both" else [args.which]
    for n in names:
        build(n, args.ffmpeg, media, work, media / f"social_{n}.mp4")


if __name__ == "__main__":
    main()
