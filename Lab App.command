#!/bin/zsh
# macOS: double-click to open the Reaching Lab app (lab_app.py).
here="${0:A:h}"
python="$here/venv/bin/python"
[[ -x "$python" ]] || python="${here:h}/venv/bin/python"
if [[ ! -x "$python" ]]; then
    echo "Python environment not found ($here/venv) - see 'Setup on macOS' in README.md."
    read -s -k 1 "?Press any key to close..."
    exit 1
fi
export GLOG_minloglevel=2 TF_CPP_MIN_LOG_LEVEL=3
cd "$here" && "$python" lab_app.py
