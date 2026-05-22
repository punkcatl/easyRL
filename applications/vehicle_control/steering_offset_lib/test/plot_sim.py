#!/usr/bin/env python3
"""Plot steering offset filter simulation results."""

import os
import sys
import subprocess
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 150,
})


def _detect_cjk_font():
    """Try to find a usable CJK font. Returns FontProperties or None."""
    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
    ]
    for path in candidates:
        try:
            prop = fm.FontProperties(fname=path)
            if prop.get_file():
                return prop
        except Exception:
            continue

    for f in fm.findSystemFonts():
        if "CJK" in f and "SC" in f and f.endswith((".ttf", ".otf", ".ttc")):
            try:
                return fm.FontProperties(fname=f)
            except Exception:
                continue

    try:
        prop = fm.FontProperties(family="Noto Sans CJK SC")
        if fm.findfont(prop) != fm.findfont("DejaVu Sans"):
            return prop
    except Exception:
        pass

    return None


_cjk_font = _detect_cjk_font()
_use_chinese = _cjk_font is not None

if _use_chinese:
    plt.rcParams.update({
        "axes.unicode_minus": False,
    })

# Bilingual label map
_L = {
    "suptitle":       ("Steering Offset Estimation Simulation (injected bias = 10 deg)",
                       "方向盘零偏估计仿真验证（注入零偏 = 10°）"),
    "sp1_title":      ("Vehicle Speed Profile", "车速剖面"),
    "sp1_ylabel":     ("Velocity (m/s)", "车速 (m/s)"),
    "sp1_legend":     ("velocity_threshold (5 m/s)", "速度阈值 (5 m/s)"),
    "sp2_title":      ("Offset Estimate Convergence", "零偏估计收敛过程"),
    "sp2_ylabel":     ("Wheel Angle Bias (mrad)", "前轮转角零偏 (mrad)"),
    "sp2_est":        ("KF estimate", "KF 估计值"),
    "sp2_true":       ("true bias", "真值"),
    "sp3_title":      ("Covariance Convergence & Output Enable", "协方差收敛 & 补偿启用"),
    "sp3_ylabel":     ("Variance P (rad^2)", "方差 P (rad²)"),
    "sp3_P":          ("P (variance)", "协方差 P"),
    "sp3_thresh":     ("var_valid_threshold", "收敛阈值"),
    "sp3_enabled":    ("output_enabled ON", "补偿输出启用"),
    "sp4_title":      ("Update Gate Status & Yawrate", "更新门控状态 & 横摆角速度"),
    "sp4_ylabel_l":   ("Gate Status", "Gate 状态"),
    "sp4_ylabel_r":   ("Yawrate (deg/s)", "横摆角速度 (deg/s)"),
    "sp4_xlabel":     ("Time (s)", "时间 (s)"),
    "sp4_blocked":    ("Blocked", "阻断"),
    "sp4_pass":       ("Pass", "通过"),
    "sp4_yr_meas":    ("yawrate measured", "yawrate 测量"),
    "sp4_yr_true":    ("yawrate true", "yawrate 真值"),
    "gate_frozen":    ("KF frozen (gate blocked)", "KF 冻结（gate 阻断）"),
}


def T(key):
    """Return label in the appropriate language."""
    en, zh = _L[key]
    return zh if _use_chinese else en


def _text_kwargs():
    """Extra kwargs for text elements when using CJK font."""
    if _use_chinese:
        return {"fontproperties": _cjk_font}
    return {}


def shade_gate_blocked(ax, t, gate):
    """Shade gate-blocked intervals with translucent orange."""
    in_block = False
    start = 0.0
    for i in range(len(gate)):
        if gate[i] < 0.5 and not in_block:
            start = t[i]
            in_block = True
        elif gate[i] > 0.5 and in_block:
            ax.axvspan(start, t[i], alpha=0.08, color="orange")
            in_block = False
    if in_block:
        ax.axvspan(start, t[-1], alpha=0.08, color="orange")


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "sim_output.csv"
    data = np.genfromtxt(csv_path, delimiter=",", names=True)

    t = data["time"]
    vel = data["vel_ego"]
    x_est = data["x_estimate"]
    P = data["P"]
    gate = data["gate_pass"]
    output_en = data["output_enabled"]
    valid_dur = data["valid_duration"]
    yr_true = data["yawrate_true"]
    yr_noisy = data["yawrate_noisy"]
    wa_rate = data["wheel_angle_rate"]

    SR = 13.5
    bias_true_wheel = 10.0 / 57.3 / SR

    kw = _text_kwargs()
    fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True)
    fig.suptitle(T("suptitle"), fontsize=13, fontweight="bold", y=0.98, **kw)

    freeze_patch = mpatches.Patch(color="orange", alpha=0.2, label=T("gate_frozen"))

    # --- Subplot 1: Velocity ---
    ax = axes[0]
    ax.plot(t, vel, "#1f77b4", linewidth=1.3)
    ax.axhline(5.0, color="#d62728", linestyle="--", alpha=0.6, label=T("sp1_legend"))
    shade_gate_blocked(ax, t, gate)
    ax.set_ylabel(T("sp1_ylabel"), **kw)
    ax.set_title(T("sp1_title"), **kw)
    ax.legend(handles=ax.get_legend_handles_labels()[0] + [freeze_patch],
              loc="upper right", prop=_cjk_font if _use_chinese else None)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-1, 25)

    # --- Subplot 2: Offset Estimate ---
    ax = axes[1]
    ax.plot(t, x_est * 1000, "#1f77b4", linewidth=1.3, label=T("sp2_est"))
    ax.axhline(bias_true_wheel * 1000, color="#d62728", linestyle="--",
               label=f"{T('sp2_true')} = {bias_true_wheel*1000:.2f} mrad")
    shade_gate_blocked(ax, t, gate)
    ax.set_ylabel(T("sp2_ylabel"), **kw)
    ax.set_title(T("sp2_title"), **kw)
    ax.legend(handles=ax.get_legend_handles_labels()[0] + [freeze_patch],
              loc="right", prop=_cjk_font if _use_chinese else None)
    ax.grid(True, alpha=0.3)

    # --- Subplot 3: Covariance + output_enabled ---
    ax = axes[2]
    ax.semilogy(t, P, "#1f77b4", linewidth=1.3, label=T("sp3_P"))
    ax.axhline(2e-8, color="#d62728", linestyle="--", alpha=0.7, label=T("sp3_thresh"))
    shade_gate_blocked(ax, t, gate)

    # Shade output_enabled region
    enabled_mask = output_en > 0.5
    if np.any(enabled_mask):
        ax.fill_between(t, 0, 1, where=enabled_mask,
                        transform=ax.get_xaxis_transform(),
                        alpha=0.06, color="#2ca02c")
        enabled_patch = mpatches.Patch(color="#2ca02c", alpha=0.15, label=T("sp3_enabled"))
        # Annotate the enable moment
        enable_idx = np.argmax(enabled_mask)
        ax.annotate(f"t={t[enable_idx]:.1f}s",
                    xy=(t[enable_idx], P[enable_idx]),
                    xytext=(t[enable_idx] + 3, P[enable_idx] * 5),
                    arrowprops=dict(arrowstyle="->", color="#2ca02c"),
                    color="#2ca02c", fontsize=9, **kw)
    else:
        enabled_patch = None

    ax.set_ylabel(T("sp3_ylabel"), **kw)
    ax.set_title(T("sp3_title"), **kw)
    legend_handles = ax.get_legend_handles_labels()[0] + [freeze_patch]
    if enabled_patch:
        legend_handles.append(enabled_patch)
    ax.legend(handles=legend_handles,
              loc="upper right", prop=_cjk_font if _use_chinese else None)
    ax.grid(True, alpha=0.3)

    # --- Subplot 4: Gate status + Yawrate ---
    ax = axes[3]
    ax2 = ax.twinx()

    ax.fill_between(t, 0, gate, alpha=0.25, color="#2ca02c", step="post")
    ax.set_ylabel(T("sp4_ylabel_l"), **kw)
    ax.set_yticks([0, 1])
    ax.set_yticklabels([T("sp4_blocked"), T("sp4_pass")], **kw)
    ax.set_ylim(-0.15, 1.5)

    ax2.plot(t, yr_noisy * 57.3, color="#aec7e8", linewidth=0.4, alpha=0.7, label=T("sp4_yr_meas"))
    ax2.plot(t, yr_true * 57.3, color="#d62728", linewidth=1.0, alpha=0.9, label=T("sp4_yr_true"))
    ax2.set_ylabel(T("sp4_ylabel_r"), **kw)

    ax.set_xlabel(T("sp4_xlabel"), **kw)
    ax.set_title(T("sp4_title"), **kw)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right",
              prop=_cjk_font if _use_chinese else None)
    ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = csv_path.replace(".csv", ".png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")

    lang = "zh-CN" if _use_chinese else "en"
    print(f"Plot saved: {out_path} (lang={lang})")

    # Auto-open image if display is available
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        try:
            subprocess.Popen(["xdg-open", out_path],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main()
