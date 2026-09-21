#!/bin/bash
# T7 COVARIATE SHIFT -- re-collect the demonstrations with noise injected into the demonstrator.
#
# WHY (session 6's diagnosis, in full in STATUS.md): the fine-tuned policy reproduces the
# demonstrations' state tube (overfit control: 3.8x better than the no-motion baseline on every
# joint) and still grasps 1/20 in closed loop. One-step prediction improved 3.5x while closed-loop
# success did not move at all -- the signature of covariate shift. Every demo came from a scripted
# controller that never erred, so the data holds no recovery states. That is a DATA change.
#
# bw/manip/disturb.py kicks the ARM off the path (servo offset) and leaves the LABEL nominal, so
# every frame from a kick until the arm is back on the path pairs an off-tube state with the
# command that corrects it. sigma 0.10 rad / p 0.15 per tick chosen on a yield sweep (seeds
# 90000-90013, sigma 0.04/0.08/0.12): 8 of 8 runs still produced a clean pick AND place, while
# |q_cmd - q_state| widened where it matters -- j1 3.0 -> 5.0 mrad median, j5 2.5 -> 5.0, the two
# joints that never beat the trivial baseline in EITHER delta run.
#
# usage: bash ops/collect_noise.sh [START] [END] [WORKERS]
cd /d/bringwrench
S=${1:-200000}; E=${2:-200600}; N=${3:-3}
mkdir -p D:/bw_data/noise
seq $S 25 $((E-1)) | xargs -P $N -I{} sh -c \
  'D:/hexapod/render_venv/Scripts/python.exe scripts/collect_demos.py {} $(( {} + 25 )) --out D:/bw_data/noise --noise 0.15 --noise-sigma 0.10 > D:/bw_data/noise_lane_{}.txt 2>&1'
echo "collection done: $(ls -d D:/bw_data/noise/*_pick 2>/dev/null | wc -l) pick, $(ls -d D:/bw_data/noise/*_place 2>/dev/null | wc -l) place"
