# 方向盘零偏滤波器仿真测试

## 工程结构

```
steering_offset_lib/
├── CMakeLists.txt
├── include/controller/steering_offset/
│   └── steering_offset_filter.h        # 头文件（Config + Filter 类）
├── src/
│   └── steering_offset_filter.cpp      # 实现
├── docs/
│   ├── 2026-05-22-steering-offset-algorithm-detail.md   # 算法详细规格
│   ├── 2026-05-22-steering-offset-simplification-design.md  # 精简方案设计
│   └── legacy-steering-offset-compensation.md           # 旧系统参考
└── test/
    ├── CMakeLists.txt
    ├── sim_main.cpp       # C++ 仿真程序
    ├── plot_sim.py        # Python 绘图
    ├── run_sim.sh         # 一键运行脚本
    └── README.md          # 本文件
```

## 编译

```bash
cd src/library/steering_offset_lib
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

## 运行

```bash
cd build/test
./sim_main
python3 ../../test/plot_sim.py sim_output.csv
```

## 仿真场景

60s 仿真，注入 10° 方向盘零偏（≈ 0.013 rad 前轮转角级）：

| 时间 (s) | 阶段 | Gate 状态 |
|-----------|------|-----------|
| 0–5 | 加速 0→20 m/s | 拒绝（低速） |
| 5–30 | 高速巡航 20 m/s，直行 | 通过 |
| 30–35 | 换道（正弦 ±0.03 rad） | 拒绝（大转角/大横摆） |
| 35–55 | 高速巡航 20 m/s，直行 | 通过 |
| 55–60 | 减速 20→3 m/s | 拒绝（低速） |

## 预期结果

- Gate 打开后 ~2s 内 KF 收敛到真值（t ≈ 7s，含 min_valid_duration）
- 换道期间估计值冻结，无跳变
- 收敛后 output_enabled 保持为 true（含冻结期间）
- 最终估计误差 < 0.01 mrad

## 输出

- `sim_output.csv` — 时间序列数据
- `sim_output.png` — 4 子图（车速、零偏估计、协方差、Gate 状态）
