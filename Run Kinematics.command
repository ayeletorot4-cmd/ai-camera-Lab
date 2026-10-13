#!/bin/zsh
# macOS launcher (double-click in Finder): pick a video in a file dialog, then run extract_kinematics.py on it.
# Same steps as "Run Kinematics.bat" on Windows.

here="${0:A:h}"
parent="${here:h}"   # videos are usually in the folder above this one
python="$here/venv/bin/python"
[[ -x "$python" ]] || python="$parent/venv/bin/python"
script="$here/extract_kinematics.py"
start_dir="$parent"
[[ -d "$here/media" ]] && start_dir="$here/media"   # downloaded videos live in media/

finish() { echo; read -s -k 1 "?Press any key to close this window..."; echo; exit ${1:-0}; }

if [[ ! -x "$python" ]]; then
    echo "Python environment not found ($here/venv)."
    echo "See 'Setup on macOS' in README.md."
    finish 1
fi

video=$(osascript <<EOF 2>/dev/null
set startFolder to POSIX file "$start_dir" as alias
POSIX path of (choose file with prompt "Choose a video to analyze" of type {"public.movie"} default location startFolder)
EOF
)
if [[ -z "$video" ]]; then
    echo 'No video selected - nothing to do.'
    finish 0
fi
echo "Video: $video"

# Which limb to track?
limb=''
while [[ -z "$limb" ]]; do
    read "answer?Which limb? Type U for upper limb (arm + fingers) or L for lower limb (leg): "
    case "$answer" in
        [Uu]*) limb=upper ;;
        [Ll]*) limb=lower ;;
    esac
done
side=''
while [[ -z "$side" ]]; do
    read "answer?Which side? Type L for the subject's LEFT or R for the subject's RIGHT: "
    case "$answer" in
        [Ll]*) side=left ;;
        [Rr]*) side=right ;;
    esac
done
extra_args=(--limb "$limb" --side "$side")

# Switch events (light start / touching / light off for the 5 switches)
read "answer?Detect switch events? Type Y for yes, or press ENTER to skip: "
if [[ "$answer" == [Yy]* ]]; then
    extra_args+=(--switches)
    video_name="${video:t}"
    if "$python" -c 'import json, sys; sys.exit(0 if json.load(open(sys.argv[1])).get(sys.argv[2]) else 1)' \
            "$here/outputs/switch_settings.json" "$video_name" 2>/dev/null; then
        read "answer?Switch positions are saved for $video_name. Press ENTER to reuse them, or type R to find them again: "
        [[ "$answer" == [Rr]* ]] && extra_args+=(--select-switches)
    else
        echo 'The switches will be found automatically (about a minute), then shown numbered 1 to 5 (RIGHT to LEFT): press ENTER if correct, or follow the keys shown to fix them.'
    fi
fi

echo
echo 'Loading MediaPipe (first frames may take a few seconds)...'
# Quiet MediaPipe/TensorFlow internal log chatter (real errors still show)
export GLOG_minloglevel=2
export TF_CPP_MIN_LOG_LEVEL=3
"$python" "$script" "$video" "${extra_args[@]}"
code=$?
if (( code != 0 )); then
    print -P "%F{red}Processing failed (exit code $code) - see the message above.%f"
fi
finish $code
