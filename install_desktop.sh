#!/usr/bin/env bash
# Put GemRadio in the applications menu (and so on the dock/toolbar), or
# take it back out again with --uninstall.
#
# Everything is installed under $HOME: nothing touches system directories and
# nothing needs root.

set -euo pipefail

HERE="$(dirname "$(readlink -f "$0")")"
APPS="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
ICONS="${XDG_DATA_HOME:-$HOME/.local/share}/icons/hicolor/scalable/apps"
DESKTOP="$APPS/gemradio.desktop"
ICON="$ICONS/gemradio.svg"

refresh() {
    command -v update-desktop-database >/dev/null 2>&1 && \
        update-desktop-database "$APPS" >/dev/null 2>&1 || true
    command -v gtk-update-icon-cache >/dev/null 2>&1 && \
        gtk-update-icon-cache -f -t "${XDG_DATA_HOME:-$HOME/.local/share}/icons/hicolor" \
        >/dev/null 2>&1 || true
}

if [[ "${1:-}" == "--uninstall" ]]; then
    rm -f "$DESKTOP" "$ICON"
    refresh
    echo "GemRadio removed from the applications menu."
    exit 0
fi

mkdir -p "$APPS" "$ICONS"
install -m 644 "$HERE/gemradio/assets/gemradio.svg" "$ICON"

cat > "$DESKTOP" <<EOF
[Desktop Entry]
Type=Application
Version=1.0
Name=GemRadio
GenericName=Offline AI Radio
Comment=Your own music, introduced by a local AI DJ. Works completely offline.
Exec=$HERE/run_gemradio.sh
Path=$HERE
Icon=gemradio
Terminal=false
Categories=AudioVideo;Audio;Player;
Keywords=radio;music;offline;ai;dj;gemma;whisper;piper;
StartupNotify=true
StartupWMClass=gemradio
EOF
chmod 644 "$DESKTOP"
refresh

echo "GemRadio installed in the applications menu."
echo "  entry: $DESKTOP"
echo "  icon:  $ICON"
echo
echo "Find it in Activities as 'GemRadio'; right-click its icon and choose"
echo "'Pin to Dash' to keep it on the dock."
