#!/bin/zsh

set -euo pipefail

PROJECT_ROOT="${0:A:h:h}"
PYTHON_BIN="${HGASTRO_PYTHON:-/Volumes/SSDAPFS/conda/envs/hgastro/bin/python}"
APP_NAME="HoshinoPanoAssistant"
APP_BUNDLE="$PROJECT_ROOT/dist/$APP_NAME.app"
RELEASE_DIR="$PROJECT_ROOT/dist/$APP_NAME-macOS-arm64"
RELEASE_APP="$RELEASE_DIR/$APP_NAME.app"
ICON_SOURCE="$PROJECT_ROOT/icon256.png"
ICON_OUTPUT="$PROJECT_ROOT/build/$APP_NAME.icns"
METDET_DIR="${METDET_WORKER_DIR:-$PROJECT_ROOT/../MetDetPy/dist/metdet_worker}"

if [[ ! -x "$PYTHON_BIN" ]]; then
    print -u2 "找不到 hgastro Python：$PYTHON_BIN"
    exit 1
fi

if [[ ! -f "$ICON_SOURCE" ]]; then
    print -u2 "找不到应用图标：$ICON_SOURCE"
    exit 1
fi

if [[ ! -d "$PROJECT_ROOT/qrcode" ]]; then
    print -u2 "找不到二维码目录：$PROJECT_ROOT/qrcode"
    exit 1
fi

if [[ ! -x "$METDET_DIR/metdet_worker" ]]; then
    print -u2 "找不到可执行的 metdet_worker：$METDET_DIR/metdet_worker"
    print -u2 "可用 METDET_WORKER_DIR=/path/to/metdet_worker 覆盖默认位置。"
    exit 1
fi

cd "$PROJECT_ROOT"

"$PYTHON_BIN" -c "from PIL import Image; from pathlib import Path; src=Image.open('icon256.png').convert('RGBA'); Path('build').mkdir(exist_ok=True); src.save('build/$APP_NAME.icns', format='ICNS', sizes=[(16,16),(32,32),(64,64),(128,128),(256,256),(512,512),(1024,1024)])"

PYTHONNOUSERSITE=1 \
PYTHONPATH= \
PYINSTALLER_CONFIG_DIR="$PROJECT_ROOT/.pyinstaller-cache" \
MPLCONFIGDIR="$PROJECT_ROOT/.matplotlib-cache" \
"$PYTHON_BIN" -m PyInstaller \
    --noconfirm \
    --clean \
    --windowed \
    --name "$APP_NAME" \
    --icon "$ICON_OUTPUT" \
    --additional-hooks-dir "$PROJECT_ROOT/hooks" \
    --add-data "$PROJECT_ROOT/catalog:catalog" \
    --add-data "$PROJECT_ROOT/qrcode:qrcode" \
    --add-data "$METDET_DIR:metdet_worker" \
    --exclude-module pytest \
    "$PROJECT_ROOT/main.py"

# PyInstaller 会先签名主应用；所有附加资源就位后再次签名，避免后续复制资源破坏 seal。
codesign --force --deep --sign - "$APP_BUNDLE"
codesign --verify --deep --strict "$APP_BUNDLE"

mkdir -p "$RELEASE_DIR"
rm -rf "$RELEASE_APP"
ditto "$APP_BUNDLE" "$RELEASE_APP"
cp "$PROJECT_ROOT/preference.json" "$RELEASE_DIR/preference.json"

codesign --verify --deep --strict "$RELEASE_APP"
print "构建完成：$RELEASE_APP"
