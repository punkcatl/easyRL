#!/bin/bash
# 方向盘零偏滤波器仿真 — 一键运行脚本
#
# 用法:
#   ./run_sim.sh          # 编译 + 仿真 + 绘图（默认全部）
#   ./run_sim.sh build    # 仅编译
#   ./run_sim.sh sim      # 仅运行仿真（需已编译）
#   ./run_sim.sh plot     # 仅绘图（需已有 CSV）
#   ./run_sim.sh sim plot # 仿真 + 绘图
#   ./run_sim.sh clean    # 清除 build 目录

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LIB_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$LIB_DIR/build"
TEST_BUILD_DIR="$BUILD_DIR/test"

do_build() {
    echo "=== 编译 ==="
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"
    cmake .. -DCMAKE_BUILD_TYPE=Release
    make -j"$(nproc)"
    echo "编译完成"
}

do_sim() {
    echo "=== 运行仿真 ==="
    cd "$TEST_BUILD_DIR"
    ./sim_main
}

do_plot() {
    echo "=== 绘图 ==="
    cd "$TEST_BUILD_DIR"
    python3 "$SCRIPT_DIR/plot_sim.py" sim_output.csv
}

do_clean() {
    echo "=== 清除 build ==="
    rm -rf "$BUILD_DIR"
    echo "已清除: $BUILD_DIR"
}

# 无参数时执行全部步骤
if [ $# -eq 0 ]; then
    do_build
    do_sim
    do_plot
    exit 0
fi

# 按参数顺序执行
for arg in "$@"; do
    case "$arg" in
        build) do_build ;;
        sim)   do_sim ;;
        plot)  do_plot ;;
        clean) do_clean ;;
        *)
            echo "未知选项: $arg"
            echo "可用: build, sim, plot, clean"
            exit 1
            ;;
    esac
done
