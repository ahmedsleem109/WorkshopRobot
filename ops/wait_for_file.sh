#!/bin/bash
# Wait until $1 exists, or until no process matching $2 is left.
# A script file, not an inline wsl.exe one-liner: multi-line shell through
# Git Bash -> wsl.exe -> bash -lc gets mangled (measured repeatedly 2026-09-18).
TARGET=$1
PROC=$2
LIMIT=${3:-120}
i=0
while [ "$i" -lt "$LIMIT" ]; do
    if [ -f "$TARGET" ]; then
        echo "READY $TARGET"
        ls -la "$TARGET"
        exit 0
    fi
    if ! pgrep -f "$PROC" > /dev/null; then
        echo "PROCESS GONE before $TARGET appeared"
        exit 1
    fi
    sleep 20
    i=$((i + 1))
done
echo "TIMEOUT waiting for $TARGET"
exit 2
