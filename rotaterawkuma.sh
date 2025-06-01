#!/usr/bin/env bash
#
# rotate-rawkuma.sh
# ---------------------------------------------------------------------------
# 1) Ask for a directory (default = your Windows “Downloads” folder)
# 2) Feed every translated_rawkuma*.png below that dir into fzf
#    – all files are pre-selected
# 3) Run rotate_image.py on each chosen file
#
# Requires:
#   • Git-Bash / WSL / any Bash with fzf installed
#   • rotate_image.py in $PATH   (or adjust ROTATE_PY below)
# ---------------------------------------------------------------------------

set -euo pipefail

# Where’s the Python rotator?
ROTATE_PY="rotateimage.py"

# 1) default = Windows “Downloads” if USERPROFILE exists,
#              else fallback to ~/Downloads
if [[ -n "${USERPROFILE:-}" ]]; then
  default_dl="${USERPROFILE//\\//}/Downloads"
else
  default_dl="$HOME/Downloads"
fi

# prompt (read -e = readline editing, -r = raw, -p = prompt text)
read -e -r -p "Directory to scan [${default_dl}]: " scan_dir
scan_dir=${scan_dir:-$default_dl}

# expand a leading ~, convert back-slashes → forward-slashes
scan_dir="${scan_dir/#\~/$HOME}"
scan_dir=$(printf '%s' "$scan_dir" | tr '\\' '/')

if [[ ! -d "$scan_dir" ]]; then
  printf '❌  "%s" is not a directory.\n' "$scan_dir" >&2
  exit 1
fi
printf '📂  Scanning: %s\n' "$scan_dir"

# ---------------------------------------------------------------------------
# 1) find matching images, NUL-separated
find "$scan_dir" -type f -name 'translated_rawkuma*.png' -print0 |
# 2) fzf:   multi-select, read0/print0, start with all selected
fzf --read0 --print0 -m \
    --bind 'start:select-all,ctrl-a:select-all,ctrl-d:deselect-all,ctrl-t:toggle-all' \
    --prompt='[rotate ⇧⏎ to run] > ' |
# 3) loop over selections (still NUL-safe)

# ── run the rotator in parallel ────────────────────────────────
#    -0           : NUL-terminated input from fzf
#    -n1          : feed one filename per invocation
#    -P <N>       : number of parallel jobs (here = CPU cores)
#    -I{} … {}    : safely substitute the filename
xargs -0 -n1 -P "$(nproc)" -I{} \
  bash -c 'printf "↻  %s\n" "{}"; python "'"$ROTATE_PY"'" "{}"'