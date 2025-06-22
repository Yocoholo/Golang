#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(dirname $0)"
OUTPUT_DIR_BASE="$SCRIPT_DIR/out"
OUTPUT_DIR=""
BUILD=""
EXPORT_DIR_NAME=""

show_usage() {
    echo "Usage: $0 [NAME] [OPTIONS]"
    echo "  NAME: the name of the dir youd like to create"]
    echo ""
    echo "  Example: $0 nnlcpp"
}

if [ "$#" -lt 1 ]; then
    show_usage
    exit 1
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        -b|--build)
            BUILD="1"
            shift 1
            ;;
        -h|--help)
            echo help
            show_usage
            exit 0
            ;;
        *)
            EXPORT_DIR_NAME=$1
            shift
            ;;
    esac
done

OUTPUT_DIR="$OUTPUT_DIR_BASE/$EXPORT_DIR_NAME"

if [ -n "$BUILD" ]; then
    echo "Building in $OUTPUT_DIR"
    cmake --build $OUTPUT_DIR -j$(nproc)
    if [ $? -ne 0 ]; then
        echo "Build failed!"
        exit 1
    fi
    echo ""
    echo "Build complete!"
else
    echo "Configuring project in $OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR"

    cmake -S $SCRIPT_DIR -B $OUTPUT_DIR -G "Ninja"

    echo ""
    echo "Configuration complete!"
    echo "You can now run: $0 -b $EXPORT_DIR_NAME"
fi

