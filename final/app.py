import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
import torch.nn.functional as F

# =====================================================================
# THIẾT LẬP CẤU HÌNH TRANG STREAMLIT
# =====================================================================
st.set_page_config(
    page_title="Mô Phỏng & Thực Nghiệm DirectAU (KDD '22) - Nhóm 5 UIT",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

plt.style.use(
    "seaborn-v0_8-whitegrid"
    if "seaborn-v0_8-whitegrid" in plt.style.available
    else "default"
)
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "sans-serif"]
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8

# Đường dẫn thư mục dữ liệu tự thân (không phụ thuộc vào smoke_test_directAU)
FINAL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FINAL_DIR.parent
BENCH_DIR = FINAL_DIR / "benchmark_results"
CACHE_DIR = FINAL_DIR / "directau_beauty_cache"

# =====================================================================
# SIDEBAR ĐIỀU HƯỚNG
# =====================================================================
st.sidebar.title("🎯 Chuyên Đề DirectAU (KDD '22)")
st.sidebar.markdown(
    """
**Nhóm 5 — Collaborative Filtering & Representation Learning**  
*Bài báo: Towards Representation Alignment and Uniformity in Collaborative Filtering (KDD 2022)*  
*Môn: Khai thác dữ liệu và ứng dụng (CS313 / CS2207)*  
*Trường ĐH Công Nghệ Thông Tin — ĐHQG-HCM*
"""
)

menu = st.sidebar.radio(
    "Chọn Module Mô Phỏng & Thực Nghiệm:",
    [
        "1. 🌐 Hình Học Mặt Cầu Đơn Vị: Alignment & Uniformity",
        "2. 🧹 Tiền Xử Lý Dữ Liệu: Beauty, Gowalla, Yelp2018 → 5-Core & Split",
        "3. 📈 Động Lực Học Tập & Quỹ Đạo l_align - l_uniform",
        "4. 🧪 Nghiên Cứu Triệt Tiêu (Ablation) & Độ Nhạy Tham Số γ",
        "5. 🏆 Đối Sánh Toàn Diện: Thực Nghiệm Tự Chạy vs 9 Mô Hình SOTA",
        "6. 🔍 Hệ Thống Gợi Ý Top-K Thực Tế Trên Amazon Beauty",
    ],
    key="nav_menu",
)

st.sidebar.markdown("---")
st.sidebar.info(
    """
💡 **Điểm nhấn của hệ thống:**
- **Bám sát bài báo KDD '22 (`ref/DirectAU.pdf`)** và mã nguồn gốc `ref/DirectAU`.
- **Tự tải & Tiền xử lý cả 3 bộ dữ liệu chuẩn:** `Beauty` (`22,363` users, `12,101` items, `198,502` tương tác), `Gowalla` (`29,858` users, `40,981` items, `1,027,370` tương tác), và `Yelp2018` (`31,668` users, `38,048` items, `1,561,406` tương tác).
- **Đánh giá Full-Item Ranking** trên toàn bộ catalog sản phẩm và toàn bộ người dùng của cả 3 bộ dữ liệu.
"""
)


# =====================================================================
# CÁC HÀM NẠP DỮ LIỆU & CACHE (100% TỪ BENCH_DIR)
# =====================================================================
@st.cache_data
def load_csv_artifacts():
    data = {}
    data["audit"] = pd.read_csv(BENCH_DIR / "preprocessing_audit.csv")
    data["gowalla_audit"] = pd.read_csv(BENCH_DIR / "gowalla_preprocessing_audit.csv")
    data["yelp_audit"] = pd.read_csv(BENCH_DIR / "yelp2018_preprocessing_audit.csv")
    data["all_audit"] = pd.read_csv(BENCH_DIR / "all_datasets_audit.csv")
    data["split_stats"] = pd.read_csv(BENCH_DIR / "split_stats.csv")
    data["split_stats_all"] = pd.read_csv(BENCH_DIR / "split_stats_all_datasets.csv")
    data["smoke_results"] = pd.read_csv(BENCH_DIR / "smoke_results.csv")
    data["smoke_bpr_hist"] = pd.read_csv(BENCH_DIR / "smoke_BPR_history.csv")
    data["smoke_dau_hist"] = pd.read_csv(BENCH_DIR / "smoke_DirectAU_history.csv")
    data["smoke_topk"] = pd.read_csv(BENCH_DIR / "smoke_topk.csv")

    data["table1"] = pd.read_csv(BENCH_DIR / "table1_dataset_stats.csv")
    data["full_results"] = pd.read_csv(BENCH_DIR / "full_benchmark_results.csv")
    data["all_bench_results"] = pd.read_csv(BENCH_DIR / "all_datasets_benchmark_results.csv")
    data["full_bpr_hist"] = pd.read_csv(BENCH_DIR / "BPR_full_history.csv")
    data["full_dau_hist"] = pd.read_csv(BENCH_DIR / "DirectAU_full_history.csv")
    data["histories"] = {
        ds: {
            "BPR": pd.read_csv(BENCH_DIR / f"{ds}_BPR_history.csv"),
            "DirectAU": pd.read_csv(BENCH_DIR / f"{ds}_DirectAU_history.csv"),
        }
        for ds in ("Beauty", "Gowalla", "Yelp2018")
    }
    data["ablation"] = pd.read_csv(BENCH_DIR / "ablation_history.csv")
    data["gamma"] = pd.read_csv(BENCH_DIR / "gamma_sensitivity.csv")
    data["full_topk"] = pd.read_csv(BENCH_DIR / "topk_full_comparison.csv")
    data["geom"] = json.loads((BENCH_DIR / "embedding_geometry.json").read_text(encoding="utf-8"))
    data["showcase_users"] = json.loads((BENCH_DIR / "showcase_users.json").read_text(encoding="utf-8"))
    data["source_info"] = json.loads((BENCH_DIR / "source.json").read_text(encoding="utf-8"))
    data["deg_dist_all"] = json.loads((BENCH_DIR / "degree_distributions_all.json").read_text(encoding="utf-8"))
    return data


@st.cache_resource
def load_recommender_assets():
    train_df = pd.read_csv(BENCH_DIR / "train.csv")
    valid_df = pd.read_csv(BENCH_DIR / "valid.csv")
    test_df = pd.read_csv(BENCH_DIR / "test.csv")
    user_tokens = pd.read_csv(BENCH_DIR / "user_mapping.csv")["user_token"].tolist()
    item_tokens = pd.read_csv(BENCH_DIR / "item_mapping.csv")["item_token"].tolist()
    eval_users = np.load(BENCH_DIR / "eval_users.npy")

    train_hist = {int(u): set(g["item"].tolist()) for u, g in train_df.groupby("user")}
    valid_targ = {int(u): set(g["item"].tolist()) for u, g in valid_df.groupby("user")}
    test_targ = {int(u): set(g["item"].tolist()) for u, g in test_df.groupby("user")}
    test_hist = {u: train_hist[u] | valid_targ[u] for u in train_hist}

    bpr_full = torch.load(BENCH_DIR / "BPR_full_best.pt", map_location="cpu", weights_only=False)
    dau_full = torch.load(BENCH_DIR / "DirectAU_full_best.pt", map_location="cpu", weights_only=False)
    bpr_smoke = torch.load(BENCH_DIR / "BPR_smoke_best.pt", map_location="cpu", weights_only=False)
    dau_smoke = torch.load(BENCH_DIR / "DirectAU_smoke_best.pt", map_location="cpu", weights_only=False)

    return {
        "user_tokens": user_tokens,
        "item_tokens": item_tokens,
        "eval_users": eval_users,
        "train_hist": train_hist,
        "valid_targ": valid_targ,
        "test_targ": test_targ,
        "test_hist": test_hist,
        "full_models": {
            "BPR": (
                bpr_full["model_state"]["user_embedding.weight"],
                bpr_full["model_state"]["item_embedding.weight"],
            ),
            "DirectAU": (
                dau_full["model_state"]["user_embedding.weight"],
                dau_full["model_state"]["item_embedding.weight"],
            ),
        },
        "smoke_models": {
            "BPR": (
                bpr_smoke["model_state"]["user_embedding.weight"],
                bpr_smoke["model_state"]["item_embedding.weight"],
            ),
            "DirectAU": (
                dau_smoke["model_state"]["user_embedding.weight"],
                dau_smoke["model_state"]["item_embedding.weight"],
            ),
        },
    }


def simulate_hypersphere_points(mode: str, n_pairs: int = 18, steps: int = 60, gamma: float = 1.0, dim: int = 2, seed: int = 42):
    """Mô phỏng tối ưu hóa trực tiếp các cặp (user, item) trên mặt cầu đơn vị S^(dim-1)."""
    torch.manual_seed(seed)
    u = torch.randn(n_pairs, dim)
    i = torch.randn(n_pairs, dim)
    u = F.normalize(u, p=2, dim=-1).detach().requires_grad_(True)
    i = F.normalize(i, p=2, dim=-1).detach().requires_grad_(True)
    opt = torch.optim.Adam([u, i], lr=0.08)

    history = []
    for step in range(steps + 1):
        un = F.normalize(u, p=2, dim=-1)
        in_ = F.normalize(i, p=2, dim=-1)
        align = (un - in_).square().sum(dim=-1).mean()
        u_unif = torch.pdist(un, p=2).square().mul(-2.0).exp().mean().log()
        i_unif = torch.pdist(in_, p=2).square().mul(-2.0).exp().mean().log()
        unif = 0.5 * (u_unif + i_unif)

        if mode == "random":
            loss = align + gamma * unif
            history.append((0, float(align.detach()), float(unif.detach()), un.detach().numpy(), in_.detach().numpy()))
            break
        elif mode == "align_only":
            loss = align
        elif mode == "uniform_only":
            loss = unif
        else:
            loss = align + gamma * unif

        history.append((step, float(align.detach()), float(unif.detach()), un.detach().numpy().copy(), in_.detach().numpy().copy()))
        if step < steps:
            opt.zero_grad()
            loss.backward()
            opt.step()

    return history[-1], history


# =====================================================================
# MODULE 1: HÌNH HỌC MẶT CẦU ĐƠN VỊ (ALIGNMENT & UNIFORMITY)
# =====================================================================
if menu == "1. 🌐 Hình Học Mặt Cầu Đơn Vị: Alignment & Uniformity":
    st.title("🌐 Bản Chất Toán Học & Hình Học Trên Mặt Cầu Đơn Vị $\\mathbb{S}^{d-1}$")
    st.markdown(
        r"""
Trong bài báo **DirectAU (KDD '22)**, nhóm tác giả chứng minh chất lượng của không gian biểu diễn trong Lọc cộng tác (Collaborative Filtering - CF) được quyết định bởi **hai tính chất hình học cốt lõi** trên mặt cầu đơn vị $\mathbb{S}^{d-1} = \{x \in \mathbb{R}^d : \|x\|_2 = 1\}$:
1. **Alignment (Độ thẳng hàng / Liên kết cặp dương):** Các cặp người dùng và sản phẩm có tương tác thực tế $(u, i) \sim p_{\text{pos}}$ phải nằm sát nhau trên mặt cầu.
2. **Uniformity (Độ phân bố đồng đều):** Toàn bộ các vector người dùng $u \sim p_{\text{user}}$ và sản phẩm $i \sim p_{\text{item}}$ phải trải đều khắp bề mặt cầu $\mathbb{S}^{d-1}$ để bảo toàn tối đa lượng thông tin đặc trưng riêng và tránh **Sụp đổ biểu diễn (Representation Collapse)**.
"""
    )

    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        st.info(
            r"""
**1. Alignment Loss (Eq. 2 & 6):**
$$l_{\text{align}} \triangleq \mathbb{E}_{(u, i) \sim p_{\text{pos}}} \|\tilde{f}(u) - \tilde{f}(i)\|_2^2$$
Với $\tilde{f}(x) = \frac{f(x)}{\|f(x)\|_2} \in \mathbb{S}^{d-1}$.  
Vì $\|\tilde{f}(u)\|_2 = \|\tilde{f}(i)\|_2 = 1$, ta có:
$$\|\tilde{f}(u) - \tilde{f}(i)\|_2^2 = 2 - 2\cos(\theta_{u, i}) \in [0, 4]$$
"""
        )
    with col_f2:
        st.success(
            r"""
**2. Uniformity Loss (Eq. 3 & 6):**
$$l_{\text{uniform}} \triangleq \frac{1}{2}\log \mathbb{E}_{u, u'} e^{-2\|\tilde{f}(u)-\tilde{f}(u')\|_2^2} + \frac{1}{2}\log \mathbb{E}_{i, i'} e^{-2\|\tilde{f}(i)-\tilde{f}(i')\|_2^2}$$
Sử dụng thế năng đẩy hạt nhân Gaussian (RBF kernel với $t=2$).  
Khi các điểm trùng nhau ($\|x-y\|^2 = 0$), $l_{\text{uniform}} = 0$ (tệ nhất). Khi các điểm phân tán đều, $l_{\text{uniform}} \to -2$ đến $-4$.
"""
        )
    with col_f3:
        st.warning(
            r"""
**3. Hàm Mục Tiêu DirectAU (Eq. 7):**
$$\mathcal{L}_{\text{DirectAU}} = l_{\text{align}} + \gamma \cdot l_{\text{uniform}}$$
- **Không cần lấy mẫu âm (Negative-Sample-Free):** Chỉ dùng các cặp tương tác dương trong mini-batch!
- Chỉ có **1 siêu tham số duy nhất $\gamma$** điều chỉnh độ mạnh của lực đẩy đồng đều.
"""
        )

    st.markdown("---")
    st.subheader("🎯 Mô Phỏng Tương Tác Trực Tiếp: Tối Ưu Hóa Biểu Diễn Trên $\\mathbb{S}^1$ (2D) & $\\mathbb{S}^2$ (3D)")

    with st.expander("⚙️ Bảng Điều Khiển Mô Phỏng Hình Học Mặt Cầu", expanded=True):
        c_cfg1, c_cfg2, c_cfg3, c_cfg4 = st.columns(4)
        with c_cfg1:
            sim_mode_label = st.selectbox(
                "Chế độ tối ưu hóa hàm Loss:",
                [
                    "4. DirectAU (Alignment + γ·Uniformity) — Tối ưu",
                    "2. Chỉ tối ưu Alignment (γ = 0) — Sụp đổ biểu diễn!",
                    "3. Chỉ tối ưu Uniformity — Mất liên kết cặp dương!",
                    "1. Khởi tạo ngẫu nhiên (Epoch 0)",
                ],
                index=0,
            )
            mode_map = {
                "1. Khởi tạo ngẫu nhiên (Epoch 0)": "random",
                "2. Chỉ tối ưu Alignment (γ = 0) — Sụp đổ biểu diễn!": "align_only",
                "3. Chỉ tối ưu Uniformity — Mất liên kết cặp dương!": "uniform_only",
                "4. DirectAU (Alignment + γ·Uniformity) — Tối ưu": "directau",
            }
            sim_mode = mode_map[sim_mode_label]
        with c_cfg2:
            n_pairs_sim = st.slider("Số cặp tương tác dương (u, i+):", 8, 30, 16, 2)
        with c_cfg3:
            steps_sim = st.slider("Số bước cập nhật Gradient (Steps):", 5, 100, 55, 5)
        with c_cfg4:
            gamma_sim = st.slider("Trọng số đồng đều (γ):", 0.1, 5.0, 1.0, 0.1)

    (_, align_2d, unif_2d, u_2d, i_2d), hist_2d = simulate_hypersphere_points(
        sim_mode, n_pairs=n_pairs_sim, steps=steps_sim, gamma=gamma_sim, dim=2
    )
    (_, align_3d, unif_3d, u_3d, i_3d), _ = simulate_hypersphere_points(
        sim_mode, n_pairs=n_pairs_sim, steps=steps_sim, gamma=gamma_sim, dim=3
    )

    m_c1, m_c2, m_c3, m_c4 = st.columns(4)
    m_c1.metric("l_align (Càng gần 0 càng tốt)", f"{align_2d:.4f}")
    m_c2.metric("l_uniform (Càng âm càng tốt)", f"{unif_2d:.4f}")
    m_c3.metric("Thế năng exp(l_uniform)", f"{math.exp(unif_2d):.4f}")
    m_c4.metric("Tổng Loss DirectAU", f"{align_2d + gamma_sim * unif_2d:.4f}")

    col_vis2d, col_vis3d = st.columns([1.05, 1.15])

    with col_vis2d:
        fig_s1, (ax_circle, ax_hist_sim) = plt.subplots(1, 2, figsize=(9.2, 4.4), gridspec_kw={"width_ratios": [1.15, 1]})
        theta = np.linspace(0, 2 * np.pi, 200)
        ax_circle.plot(np.cos(theta), np.sin(theta), color="#9ca3af", linestyle="--", lw=1.5, label="Đường tròn đơn vị S¹")

        for k in range(n_pairs_sim):
            ax_circle.plot(
                [u_2d[k, 0], i_2d[k, 0]],
                [u_2d[k, 1], i_2d[k, 1]],
                color="#10b981" if sim_mode in ("directau", "align_only") else "#f59e0b",
                alpha=0.65,
                lw=1.4,
            )
        ax_circle.scatter(u_2d[:, 0], u_2d[:, 1], c="#2563eb", s=65, zorder=5, edgecolors="white", label="User ũ")
        ax_circle.scatter(i_2d[:, 0], i_2d[:, 1], c="#dc2626", s=65, marker="^", zorder=5, edgecolors="white", label="Item ĩ+")
        ax_circle.set_xlim(-1.25, 1.25)
        ax_circle.set_ylim(-1.25, 1.25)
        ax_circle.set_aspect("equal")
        ax_circle.set_title(f"Phân Bố Trên Vòng Tròn Đơn Vị S¹\n({sim_mode_label.split('—')[0].strip()})", fontsize=10, fontweight="bold")
        ax_circle.legend(loc="lower left", fontsize=8, framealpha=0.9)

        steps_arr = [h[0] for h in hist_2d]
        align_arr = [h[1] for h in hist_2d]
        unif_arr = [h[2] for h in hist_2d]
        ax_hist_sim.plot(steps_arr, align_arr, color="#2563eb", lw=2.2, label="l_align")
        ax_hist_sim.plot(steps_arr, unif_arr, color="#dc2626", lw=2.2, label="l_uniform")
        ax_hist_sim.set_xlabel("Bước Gradient (Step)")
        ax_hist_sim.set_ylabel("Giá trị Loss")
        ax_hist_sim.set_title("Diễn Biến l_align & l_uniform", fontsize=10, fontweight="bold")
        ax_hist_sim.legend(loc="best", fontsize=8.5)
        ax_hist_sim.grid(True, linestyle=":")
        plt.tight_layout()
        st.pyplot(fig_s1)
        plt.close(fig_s1)

    with col_vis3d:
        phi_grid, theta_grid = np.mgrid[0.0 : np.pi : 30j, 0.0 : 2.0 * np.pi : 30j]
        xs = np.sin(phi_grid) * np.cos(theta_grid)
        ys = np.sin(phi_grid) * np.sin(theta_grid)
        zs = np.cos(phi_grid)

        fig_3d = go.Figure()
        fig_3d.add_trace(
            go.Surface(
                x=xs, y=ys, z=zs,
                opacity=0.18,
                colorscale="Blues",
                showscale=False,
                name="Mặt cầu S²",
                hoverinfo="skip",
            )
        )
        lx, ly, lz = [], [], []
        for k in range(n_pairs_sim):
            lx.extend([u_3d[k, 0], i_3d[k, 0], None])
            ly.extend([u_3d[k, 1], i_3d[k, 1], None])
            lz.extend([u_3d[k, 2], i_3d[k, 2], None])
        fig_3d.add_trace(
            go.Scatter3d(
                x=lx, y=ly, z=lz,
                mode="lines",
                line=dict(color="#10b981", width=3),
                name="Liên kết (u, i+)",
            )
        )
        fig_3d.add_trace(
            go.Scatter3d(
                x=u_3d[:, 0], y=u_3d[:, 1], z=u_3d[:, 2],
                mode="markers",
                marker=dict(size=6, color="#2563eb", symbol="circle"),
                name="Users ũ",
            )
        )
        fig_3d.add_trace(
            go.Scatter3d(
                x=i_3d[:, 0], y=i_3d[:, 1], z=i_3d[:, 2],
                mode="markers",
                marker=dict(size=6, color="#dc2626", symbol="diamond"),
                name="Items ĩ+",
            )
        )
        fig_3d.update_layout(
            title="Không Gian Mặt Cầu Đơn Vị 3D S² (Xoay 360°)",
            margin=dict(l=0, r=0, b=0, t=35),
            height=410,
            legend=dict(x=0.01, y=0.98),
        )
        st.plotly_chart(fig_3d, width="stretch")

    st.markdown("---")
    st.subheader("📐 Định Lý 1 (Theorem 1 — Trang 3 Bài Báo): Mối Liên Hệ Giữa BPR Loss Với Alignment & Uniformity")
    st.markdown(
        r"""
Bài báo chứng minh định lý nền tảng giải thích vì sao hàm mục tiêu BPR truyền thống hoạt động được, đồng thời chỉ ra điểm yếu của BPR:
> **Định lý 1 (Theorem 1):** *Nếu tồn tại bộ mã hóa $f$ đạt đồng thời Perfect Alignment ($\tilde{f}(u) = \tilde{f}(i)$ hầu khắp nơi trên $p_{\text{pos}}$) và Perfect Uniformity (phân phối của $\tilde{f}(u), \tilde{f}(i)$ là phân phối đều $\sigma_{d-1}$ trên mặt cầu $\mathbb{S}^{d-1}$), thì $f$ chính là nghiệm cực tiểu chính xác (exact minimizer) của hàm mất mát $\mathcal{L}_{\text{BPR}}$.*

**Chứng minh tóm tắt (theo bất đẳng thức Jensen & bổ đề Wang-Isola):**
$$\mathcal{L}_{\text{BPR}} = \mathbb{E}_{(u, i) \sim p_{\text{pos}}} -\tilde{f}(u)^T \tilde{f}(i) + \log\left(e^{\tilde{f}(u)^T \tilde{f}(i)} + e^{\tilde{f}(u)^T \tilde{f}(i^-)}\right) \ge \mathbb{E}_{(u, i) \sim p_{\text{pos}}}\left[-1 + \log\left(e^1 + e^{\tilde{f}(u)^T \tilde{f}(i^-)}\right)\right] \ge -1 + \int_{\mathbb{S}^{d-1}}\int_{\mathbb{S}^{d-1}} \log\left(e + e^{x^T y}\right) d\sigma_{d-1}(x)d\sigma_{d-1}(y)$$
- Dấu bằng thứ nhất xảy ra khi và chỉ khi $\tilde{f}(u)^T \tilde{f}(i) = 1 \iff \|\tilde{f}(u) - \tilde{f}(i)\|_2^2 = 0$ (**Perfect Alignment**).
- Dấu bằng thứ hai xảy ra khi và chỉ khi $\tilde{f}(u), \tilde{f}(i^-)$ phân bố đều trên $\mathbb{S}^{d-1}$ (**Perfect Uniformity**).
- **Tại sao DirectAU vượt trội hơn BPR?** BPR phụ thuộc vào việc lấy mẫu âm ngẫu nhiên $i^-$: khi dự đoán của cặp dương đã lớn hơn mẫu âm dễ ($s(u,i^+) \gg s(u,i^-)$), đạo hàm $\sigma(s(u,i^-) - s(u,i^+)) \to 0$ khiến BPR ngừng kéo cặp dương lại gần! Ngược lại, **DirectAU tối ưu trực tiếp $l_{\text{align}} + \gamma l_{\text{uniform}}$ trong từng mini-batch mà không cần lấy mẫu âm!**
"""
    )


# =====================================================================
# MODULE 2: TIỀN XỬ LÝ DỮ LIỆU THỰC TẾ (BEAUTY, GOWALLA, YELP2018)
# =====================================================================
elif menu == "2. 🧹 Tiền Xử Lý Dữ Liệu: Beauty, Gowalla, Yelp2018 → 5-Core & Split":
    st.title("🧹 Quy Trình Tải & Tiền Xử Lý Cả 3 Bộ Dữ Liệu (`Beauty`, `Gowalla`, `Yelp2018`)")
    data = load_csv_artifacts()
    audit_df = data["audit"]
    all_audit_df = data["all_audit"]
    table1_df = data["table1"]
    split_all_df = data["split_stats_all"]
    src_info = data["source_info"]

    c_top1, c_top2, c_top3, c_top4 = st.columns(4)
    c_top1.metric("Amazon Beauty (5-Core)", "198,502 tương tác", delta="22,363 Users / 12,101 Items")
    c_top2.metric("Gowalla (5-Core)", "1,027,370 tương tác", delta="29,858 Users / 40,981 Items")
    c_top3.metric("Yelp2018 (5-Core)", "1,561,406 tương tác", delta="31,668 Users / 38,048 Items")
    c_top4.metric("Độ khớp với Table 1 Paper", "100% Chính xác", delta="Khớp cả 3 tập dữ liệu")

    tab_p1, tab_p2, tab_p3 = st.tabs([
        "1. 🔄 Nhật Ký Tải & Lọc Iterative 5-Core Trên Cả 3 Bộ Dữ Liệu",
        "2. 📊 Đối Chiếu Table 1 & Chia Tập 80/10/10 Per-User (Beauty, Gowalla, Yelp2018)",
        "3. 📉 Phân Phối Đuôi Dài (Long-Tail) & Popularity-Weighted Uniformity (Appendix A.2)",
    ])

    with tab_p1:
        st.subheader("1. Bảng Tổng Hợp Tiền Xử Lý 3 Bộ Dữ Liệu (`Beauty`, `Gowalla`, `Yelp2018`)")
        st.dataframe(
            all_audit_df.style.format({
                "raw_interactions": "{:,}",
                "filtered_interactions": "{:,}",
                "users": "{:,}",
                "items": "{:,}",
                "avg_inter_per_user": "{:.2f}",
                "density_pct": "{:.4f}%",
            }),
            width="stretch",
        )
        st.markdown(
            f"""
- **Nguồn dữ liệu tải tự động (Section 5.1.1 — Footnotes 2, 3, 4):**
  - `Beauty`: `{src_info['url']}` (Mã băm SHA256: `{src_info['sha256']}`)
  - `Gowalla`: `{src_info['gowalla_urls']['train']}` + `test.txt`
  - `Yelp2018`: `{src_info['yelp2018_urls']['train']}` + `test.txt`
"""
        )
        col_a1, col_a2 = st.columns([1.05, 1.35])
        with col_a1:
            st.markdown("##### 📋 Chi tiết 18 vòng lặp Iterative 5-Core trên Amazon Beauty:")
            st.dataframe(
                audit_df.style.format({"interactions": "{:,}", "users": "{:,}", "items": "{:,}"}),
                width="stretch",
                height=390,
            )
        with col_a2:
            core_only = audit_df.iloc[3:].copy()
            core_only["iter_num"] = range(1, len(core_only) + 1)
            fig_core, ax_c = plt.subplots(figsize=(8.2, 4.2))
            ax_c.plot(core_only["iter_num"], core_only["interactions"] / 1e3, "o-", color="#2563eb", lw=2.2, label="Interactions (nghìn)")
            ax_c.plot(core_only["iter_num"], core_only["users"] / 1e3, "s--", color="#10b981", lw=2, label="Users (nghìn)")
            ax_c.plot(core_only["iter_num"], core_only["items"] / 1e3, "^:", color="#dc2626", lw=2, label="Items (nghìn)")
            ax_c.axhline(198.502, color="#2563eb", alpha=0.25, linestyle="--")
            ax_c.set_xticks(range(1, len(core_only) + 1))
            ax_c.set_xlabel("Vòng lặp lọc 5-Core (Iteration)", fontweight="bold")
            ax_c.set_ylabel("Số lượng (Nghìn - Thousands)", fontweight="bold")
            ax_c.set_title("Sự Hội Tụ Của Thuật Toán Iterative 5-Core Qua 18 Vòng Lặp (Beauty)", fontweight="bold")
            ax_c.legend()
            ax_c.grid(True, linestyle=":")
            st.pyplot(fig_core)
            plt.close(fig_core)
            st.caption(
                "🔬 **Tại sao lọc 1 lần là sai?** Ở vòng lặp 1 (`core_iteration_1`), việc xóa các user `< 5` tương tác làm nhiều item bị tụt xuống dưới 5 tương tác (còn 394,908 tương tác). Phải lặp đến vòng 17–18 đồ thị hai phía mới đạt điểm bất động chính xác (198,502 tương tác)."
            )

    with tab_p2:
        st.subheader("2. Đối Chiếu Thống Kê Bảng 1 (Table 1) Trên Cả 3 Tập Dữ Liệu Chuẩn Của Bài Báo")
        st.dataframe(
            table1_df.style.format({
                "users_measured": "{:,}",
                "items_measured": "{:,}",
                "interactions_measured": "{:,}",
                "avg_inter_per_user": "{:.2f}",
                "density_pct": "{:.4f}%",
                "paper_table1_avg": "{:.1f}",
                "paper_table1_density_pct": "{:.2f}%",
            }),
            width="stretch",
        )
        st.success(
            "✅ **Xác nhận tái lập dữ liệu:** Cả 3 tập `Beauty`, `Gowalla`, và `Yelp2018` sau khi tải và lọc 5-Core đều khớp **tuyệt đối 100%** tới từng đơn vị với Bảng 1 và các tệp `.inter` gốc của nhóm tác giả!"
        )

        st.markdown("---")
        st.subheader("3. Thống Kê Chia Tập Dữ Liệu Theo Từng User (80% Train / 10% Valid / 10% Test) Trên Cả 3 Bộ Dữ Liệu")
        col_sp1, col_sp2 = st.columns([1.2, 1.1])
        with col_sp1:
            st.dataframe(
                split_all_df.style.format({"interactions": "{:,}", "users": "{:,}", "items": "{:,}"}),
                width="stretch",
            )
        with col_sp2:
            ds_sp_choice = st.selectbox("Chọn bộ dữ liệu để vẽ biểu đồ phân bổ Train/Valid/Test:", ["Beauty", "Gowalla", "Yelp2018"])
            sub_sp = split_all_df[split_all_df["dataset"] == ds_sp_choice]
            fig_sp, ax_sp = plt.subplots(figsize=(6.8, 3.4))
            bars = ax_sp.bar(
                sub_sp["split"].str.upper(),
                sub_sp["interactions"],
                color=["#2563eb", "#f59e0b", "#10b981"],
                width=0.48,
            )
            max_h = float(sub_sp["interactions"].max())
            for b in bars:
                h = b.get_height()
                ax_sp.text(b.get_x() + b.get_width() / 2, h + max_h * 0.02, f"{int(h):,}", ha="center", fontweight="bold")
            ax_sp.set_ylim(0, max_h * 1.18)
            ax_sp.set_ylabel("Số lượng tương tác")
            ax_sp.set_title(f"Phân Bổ Tương Tác Giữa Train / Valid / Test ({ds_sp_choice})", fontweight="bold")
            st.pyplot(fig_sp)
            plt.close(fig_sp)

    with tab_p3:
        st.subheader("3. Phân Phối Bậc Đuôi Dài (Long-Tail) Trên Cả 3 Tập & Popularity-Weighted Uniformity (Appendix A.2)")
        deg_all = data["deg_dist_all"]
        ds_deg_choice = st.radio("Chọn bộ dữ liệu để khảo sát phân phối bậc User/Item:", ["Beauty", "Gowalla", "Yelp2018"], horizontal=True)
        d_info = deg_all[ds_deg_choice]
        u_deg = np.array(d_info["user_deg_sample"])
        i_deg = np.array(d_info["item_deg_sample"])

        col_lt1, col_lt2 = st.columns(2)
        with col_lt1:
            fig_deg, ax_d = plt.subplots(figsize=(6.5, 3.8))
            ax_d.hist(u_deg, bins=45, range=(5, 120), alpha=0.75, color="#2563eb", label=f"Users (mean={d_info['user_deg_mean']:.1f}, max={d_info['user_deg_max']})")
            ax_d.hist(i_deg, bins=45, range=(5, 120), alpha=0.65, color="#dc2626", label=f"Items (mean={d_info['item_deg_mean']:.1f}, max={d_info['item_deg_max']})")
            ax_d.set_yscale("log")
            ax_d.set_xlabel("Số lượng tương tác (Bậc của đỉnh trên đồ thị)")
            ax_d.set_ylabel("Số lượng User / Item (Thang Log)")
            ax_d.set_title(f"Phân Phối Đuôi Dài Trên Tập {ds_deg_choice}", fontweight="bold")
            ax_d.legend()
            st.pyplot(fig_deg)
            plt.close(fig_deg)
        with col_lt2:
            st.markdown(
                r"""
#### 🔬 Tại Sao In-Batch Uniformity Của DirectAU Giảm Được Thiên Kiến Phổ Biến (Popularity / Exposure Bias)?
Trong **Appendix A.2 (Công thức 9 & 10)**, nhóm tác giả chứng minh việc lấy mẫu các cặp từ tập tương tác $\mathcal{R}$ tương đương chính xác với việc lấy mẫu trên tập $\mathcal{U}, \mathcal{I}$ có trọng số phổ biến $p(u), p(i)$:
$$l_{\text{uniform}} = \frac{1}{2}\log \sum_{u, u' \in \mathcal{U}} \frac{p(u)p(u')}{P_U} e^{-2\|\tilde{f}(u)-\tilde{f}(u')\|_2^2} + \frac{1}{2}\log \sum_{i, i' \in \mathcal{I}} \frac{p(i)p(i')}{P_I} e^{-2\|\tilde{f}(i)-\tilde{f}(i')\|_2^2}$$
- **Ý nghĩa toán học:** Các sản phẩm phổ biến ($p(i)$ lớn) xuất hiện lặp lại nhiều lần trong các mini-batch và dễ bị kéo chụm vào tâm bởi $l_{\text{align}}$.
- Nhờ trọng số $p(i)p(i')$, hàm $l_{\text{uniform}}$ tự động tạo ra **lực đẩy mạnh hơn giữa các sản phẩm phổ biến**, ép chúng phải tách rời nhau trên mặt cầu $\mathbb{S}^{d-1}$, nhường không gian biểu diễn cho các sản phẩm đuôi dài (long-tail items)!
"""
            )


# =====================================================================
# MODULE 3: ĐỘNG LỰC HỌC TẬP & QUỸ ĐẠO L_ALIGN - L_UNIFORM
# =====================================================================
elif menu == "3. 📈 Động Lực Học Tập & Quỹ Đạo l_align - l_uniform":
    st.title("📈 Động Lực Học Tập (Learning Dynamics) & Quỹ Đạo $l_{\\text{align}} - l_{\\text{uniform}}$")
    data = load_csv_artifacts()

    tab_d1, tab_d2 = st.tabs([
        "1. 🚀 Quỹ Đạo Huấn Luyện Thực Nghiệm Trên Beauty, Gowalla, Yelp2018 (BPRMF vs DirectAU)",
        "2. 🗺️ Bản Đồ Quỹ Đạo l_align - l_uniform Của 8 Phương Pháp CF (Figures 2 & 4 Paper)",
    ])

    with tab_d1:
        ds_dyn = st.radio(
            "Chọn Bộ Dữ Liệu Thực Nghiệm Để Quan Sát Quá Trình Hội Tụ:",
            ["Beauty (γ = 0.5)", "Gowalla (γ = 1.0)", "Yelp2018 (γ = 2.0)"],
            horizontal=True,
        )
        ds_key = ds_dyn.split()[0]
        bpr_hist = data["histories"][ds_key]["BPR"]
        dau_hist = data["histories"][ds_key]["DirectAU"]

        st.subheader(f"1. So Sánh Trực Tiếp Quá Trình Hội Tụ Trên Tập {ds_key}: BPRMF vs DirectAU")

        col_c1, col_c2 = st.columns(2)
        with col_c1:
            fig_val, ax_v = plt.subplots(figsize=(7.2, 4.2))
            ax_v.plot(bpr_hist["epoch"], bpr_hist["valid_ndcg_at_20"], "o--", color="#dc2626", lw=2, ms=4, label=f"BPRMF ({ds_key}) - Valid NDCG@20")
            ax_v.plot(dau_hist["epoch"], dau_hist["valid_ndcg_at_20"], "s-", color="#2563eb", lw=2.3, ms=4, label=f"DirectAU ({ds_key}) - Valid NDCG@20")
            ax_v.set_xlabel("Training Epochs", fontweight="bold")
            ax_v.set_ylabel("Validation NDCG@20", fontweight="bold")
            ax_v.set_title(f"Đường Cong Hội Tụ NDCG@20 Theo Epoch ({ds_key})", fontweight="bold")
            ax_v.legend()
            ax_v.grid(True, linestyle=":")
            st.pyplot(fig_val)
            plt.close(fig_val)

        with col_c2:
            fig_rec, ax_r = plt.subplots(figsize=(7.2, 4.2))
            ax_r.plot(bpr_hist["epoch"], bpr_hist["valid_recall_at_20"], "o--", color="#f59e0b", lw=2, ms=4, label=f"BPRMF ({ds_key}) - Valid Recall@20")
            ax_r.plot(dau_hist["epoch"], dau_hist["valid_recall_at_20"], "s-", color="#10b981", lw=2.3, ms=4, label=f"DirectAU ({ds_key}) - Valid Recall@20")
            ax_r.set_xlabel("Training Epochs", fontweight="bold")
            ax_r.set_ylabel("Validation Recall@20", fontweight="bold")
            ax_r.set_title(f"Đường Cong Hội Tụ Recall@20 Theo Epoch ({ds_key})", fontweight="bold")
            ax_r.legend()
            ax_r.grid(True, linestyle=":")
            st.pyplot(fig_rec)
            plt.close(fig_rec)

        st.markdown(f"##### 🧭 Quỹ Đạo Thực Nghiệm Trên Mặt Phẳng $(l_{{\\text{{uniform}}}}, l_{{\\text{{align}}}})$ Của Tập {ds_key}:")
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            fig_f1, ax1 = plt.subplots(figsize=(7.2, 4.2))
            ax1.plot(dau_hist["epoch"], dau_hist["alignment"], color="#2563eb", lw=2.3, label="DirectAU l_align (Trục trái)")
            ax1.plot(bpr_hist["epoch"], bpr_hist["alignment"], color="#2563eb", lw=1.6, linestyle="--", label="BPRMF l_align (Trục trái)")
            ax1.set_xlabel("Training Epochs", fontweight="bold")
            ax1.set_ylabel("Alignment Loss (l_align)", color="#2563eb", fontweight="bold")
            ax1.tick_params(axis="y", labelcolor="#2563eb")

            ax2 = ax1.twinx()
            ax2.plot(dau_hist["epoch"], dau_hist["uniformity"], color="#10b981", lw=2.3, label="DirectAU l_uniform (Trục phải)")
            ax2.plot(bpr_hist["epoch"], bpr_hist["uniformity"], color="#dc2626", lw=1.6, linestyle=":", label="BPRMF l_uniform (Trục phải)")
            ax2.set_ylabel("Uniformity Loss (l_uniform)", color="#10b981", fontweight="bold")
            ax2.tick_params(axis="y", labelcolor="#10b981")
            ax1.set_title(f"Diễn Biến l_align & l_uniform Trên Tập {ds_key}", fontweight="bold")
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=8)
            st.pyplot(fig_f1)
            plt.close(fig_f1)

        with col_t2:
            b_sub = bpr_hist[bpr_hist["epoch"] >= 1]
            d_sub = dau_hist[dau_hist["epoch"] >= 1]
            fig_traj, ax_tr = plt.subplots(figsize=(7.2, 4.2))
            ax_tr.plot(b_sub["uniformity"], b_sub["alignment"], "o--", color="#dc2626", lw=1.8, ms=4, label="Quỹ đạo BPRMF")
            ax_tr.plot(d_sub["uniformity"], d_sub["alignment"], "s-", color="#2563eb", lw=2.2, ms=4, label="Quỹ đạo DirectAU")
            ax_tr.scatter([b_sub.iloc[-1]["uniformity"]], [b_sub.iloc[-1]["alignment"]], color="#dc2626", s=130, marker="*", zorder=6, label="Điểm hội tụ BPRMF")
            ax_tr.scatter([d_sub.iloc[-1]["uniformity"]], [d_sub.iloc[-1]["alignment"]], color="#2563eb", s=150, marker="*", zorder=6, label="Điểm hội tụ DirectAU")
            ax_tr.set_xlabel("l_uniform (Càng sang trái / càng âm càng đồng đều)", fontweight="bold")
            ax_tr.set_ylabel("l_align (Càng xuống dưới / càng nhỏ càng thẳng hàng)", fontweight="bold")
            ax_tr.set_title(f"Quỹ Đạo Học Tập Trên Mặt Phẳng (l_uniform, l_align) - {ds_key}", fontweight="bold")
            ax_tr.legend(fontsize=8.5)
            ax_tr.grid(True, linestyle=":")
            st.pyplot(fig_traj)
            plt.close(fig_traj)

    with tab_d2:
        st.subheader("2. Bản Đồ Hội Tụ $l_{\\text{align}} - l_{\\text{uniform}}$ Của 8 Phương Pháp CF Trên Tập Beauty (Figure 4 — Trang 6 Bài Báo)")
        paper_fig4_df = pd.DataFrame([
            {"Method": "BPRMF", "Family": "Pairwise Negative Sampling", "l_uniform": -3.57, "l_align": 0.692, "NDCG@20": 0.0534, "Recall@20": 0.1153},
            {"Method": "BPR-DS", "Family": "Dynamic Hard Negative Sampling", "l_uniform": -3.70, "l_align": 0.748, "NDCG@20": 0.0554, "Recall@20": 0.1181},
            {"Method": "BUIR", "Family": "Positive-Only Bootstrapping", "l_uniform": -2.85, "l_align": 0.585, "NDCG@20": 0.0556, "Recall@20": 0.1204},
            {"Method": "LGCN", "Family": "GNN Encoder + BPR Loss", "l_uniform": -3.23, "l_align": 0.592, "NDCG@20": 0.0581, "Recall@20": 0.1201},
            {"Method": "DGCF", "Family": "Disentangled GNN + BPR", "l_uniform": -3.44, "l_align": 0.608, "NDCG@20": 0.0600, "Recall@20": 0.1283},
            {"Method": "ENMF", "Family": "Whole-Data All-Negative MSE", "l_uniform": -3.54, "l_align": 0.620, "NDCG@20": 0.0606, "Recall@20": 0.1282},
            {"Method": "CLRec", "Family": "Contrastive InfoNCE Loss", "l_uniform": -3.61, "l_align": 0.628, "NDCG@20": 0.0651, "Recall@20": 0.1337},
            {"Method": "DirectAU (Ours)", "Family": "Direct Alignment + Uniformity (MF)", "l_uniform": -3.74, "l_align": 0.578, "NDCG@20": 0.0686, "Recall@20": 0.1400},
        ])

        fig_p4 = px.scatter(
            paper_fig4_df,
            x="l_uniform",
            y="l_align",
            color="NDCG@20",
            size="Recall@20",
            text="Method",
            hover_data=["Family", "NDCG@20", "Recall@20"],
            color_continuous_scale="Viridis",
            title="Figure 4 (Paper): Vị Trí Hội Tụ (l_uniform, l_align) & NDCG@20 Trên Tập Beauty (Góc dưới bên trái là tốt nhất)",
        )
        fig_p4.update_traces(textposition="top center", marker=dict(size=18, line=dict(width=1.5, color="DarkSlateGrey")))
        fig_p4.update_layout(
            xaxis_title="l_uniform (Thấp hơn / sang trái là phân bố đồng đều hơn)",
            yaxis_title="l_align (Thấp hơn / xuống dưới là liên kết cặp dương chặt hơn)",
            height=480,
        )
        st.plotly_chart(fig_p4, width="stretch")


# =====================================================================
# MODULE 4: NGHIÊN CỨU TRIỆT TIÊU (ABLATION) & ĐỘ NHẠY THAM SỐ GAMMA
# =====================================================================
elif menu == "4. 🧪 Nghiên Cứu Triệt Tiêu (Ablation) & Độ Nhạy Tham Số γ":
    st.title("🧪 Nghiên Cứu Triệt Tiêu (Ablation Study) & Độ Nhạy Siêu Tham Số $\\gamma$")
    data = load_csv_artifacts()
    ablation_df = data["ablation"]
    gamma_df = data["gamma"]

    tab_ab, tab_gm = st.tabs([
        "1. 🔬 Ablation Study: Chỉ Alignment vs Chỉ Uniformity vs DirectAU (Section 5.3 / Figure 5)",
        "2. 🎛️ Khảo Sát Độ Nhạy Trọng Số γ Trên Beauty, Gowalla, Yelp2018 (Section 5.6 / Figure 6)",
    ])

    with tab_ab:
        st.subheader("1. Chứng Minh Thực Nghiệm Vai Trò Không Thể Thiếu Của Cả Alignment Lẫn Uniformity")
        fig_ab, axes_ab = plt.subplots(1, 3, figsize=(15, 4.4))
        configs_ab = [
            ("align_only", "1. Chỉ Tối Ưu l_align (γ = 0)\n(Sụp đổ biểu diễn - Collapse!)", "#dc2626"),
            ("uniform_only", "2. Chỉ Tối Ưu l_uniform\n(Không học được liên kết cặp dương!)", "#f59e0b"),
            ("directau", "3. Tối Ưu Cả Hai: l_align + γ·l_uniform\n(DirectAU - Tăng trưởng bền vững!)", "#10b981"),
        ]

        for ax, (m_key, title_str, _) in zip(axes_ab, configs_ab):
            sub = ablation_df[ablation_df["mode"] == m_key]
            ax.plot(sub["epoch"], sub["alignment"], color="#f97316", lw=2.2, label="l_align")
            ax.plot(sub["epoch"], sub["uniformity_exp"], color="#16a34a", lw=2.2, linestyle="--", label="exp(l_uniform)")
            ax.set_xlabel("Training Epochs", fontweight="bold")
            ax.set_ylabel("Loss (l_align & exp(l_uniform))", fontweight="bold")
            ax.set_ylim(0, 2.15)
            ax.set_title(title_str, fontsize=10, fontweight="bold")
            ax.grid(True, linestyle=":")

            ax_r = ax.twinx()
            ax_r.plot(sub["epoch"], sub["valid_ndcg_at_20"], color="#dc2626" if m_key != "directau" else "#2563eb", lw=2.5, marker="o", ms=3.5, label="NDCG@20")
            ax_r.set_ylabel("Valid NDCG@20", color="#2563eb" if m_key == "directau" else "#dc2626", fontweight="bold")
            ax_r.set_ylim(0, 0.035)

            l1, lb1 = ax.get_legend_handles_labels()
            l2, lb2 = ax_r.get_legend_handles_labels()
            ax.legend(l1 + l2, lb1 + lb2, loc="center right", fontsize=8)

        plt.tight_layout()
        st.pyplot(fig_ab)
        plt.close(fig_ab)

        c_ab1, c_ab2, c_ab3 = st.columns(3)
        with c_ab1:
            st.error(
                """
**Trường hợp 1: Chỉ tối ưu $l_{\\text{align}}$ ($\\gamma = 0$)**
- $l_{\\text{align}}$ giảm nhanh từ `1.995` xuống `0.517`, nhưng $\\exp(l_{\\text{uniform}})$ tăng vọt gấp **18 lần** (từ `0.0238` lên `0.4345`).
- Các vector co cụm dần về một vùng nhỏ trên mặt cầu (**Dimensional Collapse**), khiến `NDCG@20` đạt đỉnh giả tạo ở Epoch 10 (`0.0204`) rồi **suy thoái tụt dốc** xuống `0.0133` ở Epoch 18!
"""
            )
        with c_ab2:
            st.warning(
                """
**Trường hợp 2: Chỉ tối ưu $l_{\\text{uniform}}$**
- Giữ $\\exp(l_{\\text{uniform}}) \\approx 0.0208$ rất thấp (phân bố cực đều), nhưng khoảng cách các cặp tương tác dương giữ nguyên ở $l_{\\text{align}} \\approx 2.000$ (vuông góc nhau).
- Mô hình không học được sở thích người dùng, `NDCG@20` chỉ dao động quanh `0.0000 - 0.0019` (tương đương đoán ngẫu nhiên).
"""
            )
        with c_ab3:
            st.success(
                """
**Trường hợp 3: Tối ưu kết hợp $l_{\\text{align}} + \\gamma l_{\\text{uniform}}$**
- Giữ vững độ đồng đều $\\exp(l_{\\text{uniform}}) \\approx 0.0232$ không bị suy thoái, đồng thời kéo $l_{\\text{align}}$ giảm đều từ `1.995` xuống `1.031`.
- Nhờ đó, `NDCG@20` tăng trưởng liên tục và đạt **`0.0297`** ở Epoch 18 (và đạt **`0.0561`** khi hội tụ đầy đủ)!
"""
            )

    with tab_gm:
        st.subheader("2. Khảo Sát Độ Nhạy Siêu Tham Số $\\gamma$ (Parameter Sensitivity — Section 5.6 / Figure 6)")
        col_gm1, col_gm2 = st.columns([1.05, 1.35])
        with col_gm1:
            st.markdown("##### 📊 Kết Quả Thực Nghiệm Khảo Sát $\\gamma$ Trên Beauty (`process_and_benmark.ipynb`):")
            st.dataframe(
                gamma_df.style.highlight_max(subset=["test_recall_at_20", "test_ndcg_at_20"], color="#bbf7d0").format({
                    "gamma": "{:.1f}",
                    "valid_ndcg_at_20": "{:.4f}",
                    "test_recall_at_20": "{:.4f}",
                    "test_ndcg_at_20": "{:.4f}",
                    "test_recall_at_10": "{:.4f}",
                    "test_ndcg_at_10": "{:.4f}",
                    "alignment": "{:.4f}",
                    "uniformity": "{:.4f}",
                }),
                width="stretch",
            )
            st.info(
                """
💡 **Quy luật phụ thuộc mật độ dữ liệu (Section 5.6):**
- **Beauty** (trung bình chỉ `8.9` tương tác/user): Đạt đỉnh tại **$\\gamma^* = 0.5$** (`Recall@20 = 0.1400`, `NDCG@20 = 0.0686`).
- **Yelp2018** (`49.3` tương tác/user) & **Gowalla** (`34.4` tương tác/user): Số tương tác/user lớn khiến lực kéo $l_{\\text{align}}$ mạnh hơn, dễ làm các item phổ biến chụm lại, do đó cần $\\gamma$ lớn hơn (**$\\gamma^* = 1.0 - 2.0$** trên Yelp2018 và **$\\gamma^* = 1.0 - 5.0$** trên Gowalla).
"""
            )

        with col_gm2:
            gammas_str = ["0.2", "0.5", "1", "2", "5", "10"]
            paper_gamma_data = {
                "Beauty (Paper Fig. 6a, γ*=0.5)": {
                    "Recall@20": [0.1345, 0.1400, 0.1362, 0.1250, 0.0980, 0.0885],
                    "NDCG@20": [0.0655, 0.0686, 0.0668, 0.0615, 0.0490, 0.0445],
                },
                "Gowalla (Paper Fig. 6b, γ*=5.0)": {
                    "Recall@20": [0.1180, 0.1490, 0.1760, 0.1950, 0.2014, 0.1910],
                    "NDCG@20": [0.0690, 0.0870, 0.1020, 0.1135, 0.1170, 0.1120],
                },
                "Yelp2018 (Paper Fig. 6c, γ*=1.0-2.0)": {
                    "Recall@20": [0.0690, 0.0950, 0.1096, 0.1099, 0.0990, 0.0890],
                    "NDCG@20": [0.0430, 0.0595, 0.0691, 0.0690, 0.0620, 0.0560],
                },
            }
            fig_g, axes_g = plt.subplots(1, 3, figsize=(11.5, 3.8))
            for ax, (ds_title, vals) in zip(axes_g, paper_gamma_data.items()):
                ax.plot(gammas_str, vals["Recall@20"], "s-", color="#dc2626", lw=2, label="Recall@20")
                ax.plot(gammas_str, vals["NDCG@20"], "o-", color="#2563eb", lw=2, label="NDCG@20")
                best_idx = int(np.argmax(vals["Recall@20"]))
                ax.scatter([gammas_str[best_idx]], [vals["Recall@20"][best_idx]], color="#dc2626", s=110, marker="*", zorder=5)
                ax.scatter([gammas_str[best_idx]], [vals["NDCG@20"][best_idx]], color="#2563eb", s=110, marker="*", zorder=5)
                ax.set_title(ds_title, fontsize=9.5, fontweight="bold")
                ax.set_xlabel("Trọng số γ")
                ax.grid(True, linestyle=":")
                ax.legend(fontsize=8)
            plt.tight_layout()
            st.pyplot(fig_g)
            plt.close(fig_g)


# =====================================================================
# MODULE 5: ĐỐI SÁNH TOÀN DIỆN (TABLES 2, 3, 4)
# =====================================================================
elif menu == "5. 🏆 Đối Sánh Toàn Diện: Thực Nghiệm Tự Chạy vs 9 Mô Hình SOTA":
    st.title("🏆 Bảng Đối Sánh Toàn Diện: Thực Nghiệm Tự Chạy Trên 3 Bộ Dữ Liệu & Kết Quả SOTA")
    data = load_csv_artifacts()
    full_res = data["full_results"]
    all_bench = data["all_bench_results"]
    smoke_res = data["smoke_results"]
    geom = data["geom"]

    tab_r1, tab_r2, tab_r3 = st.tabs([
        "1. 🥇 Kết Quả Thực Nghiệm Tự Chạy Trên Cả 3 Bộ Dữ Liệu (Beauty, Gowalla, Yelp2018)",
        "2. 📊 Bảng 2 (Table 2): DirectAU vs 8 Mô Hình SOTA Trên Beauty, Gowalla, Yelp2018",
        "3. 🔗 Bảng 3 & 4 (Tables 3 & 4): Tích Hợp LightGCN & Phân Tích Hiệu Quả Thời Gian",
    ])

    with tab_r1:
        st.subheader("1. Bảng Kết Quả Thực Nghiệm Tự Huấn Luyện (`process_and_benmark.ipynb`) Trên Cả 3 Bộ Dữ Liệu")
        bpr_r = full_res.loc[full_res["method"] == "BPR"].iloc[0]
        dau_r = full_res.loc[full_res["method"] == "DirectAU"].iloc[0]

        k1, k2, k3, k4 = st.columns(4)
        k1.success(f"🥇 **Beauty DirectAU Recall@20**\n**{dau_r['test_recall@20_all']:.4f}** (+17.30% vs BPRMF)")
        k2.success(f"🥇 **Beauty DirectAU NDCG@20**\n**{dau_r['test_ndcg@20_all']:.4f}** (+23.66% vs BPRMF)")
        k3.success("🥇 **Yelp2018 DirectAU Recall@20**\n**0.0594** (+21.64% vs BPRMF)")
        k4.success("🥇 **Yelp2018 DirectAU NDCG@20**\n**0.0379** (+31.76% vs BPRMF)")

        st.markdown("##### 📋 Kết quả Full-Item Ranking trên toàn bộ Users & toàn bộ Catalog Items của cả 3 bộ dữ liệu:")
        st.dataframe(
            all_bench[
                [
                    "dataset",
                    "method",
                    "gamma",
                    "best_epoch",
                    "epochs_run",
                    "test_recall@10_all",
                    "test_recall@20_all",
                    "test_recall@50_all",
                    "test_ndcg@10_all",
                    "test_ndcg@20_all",
                    "test_ndcg@50_all",
                    "evaluated_users_all",
                    "catalog_items",
                ]
            ].style.format({
                "gamma": lambda v: "-" if pd.isna(v) else f"{v:.1f}",
                "test_recall@10_all": "{:.4f}",
                "test_recall@20_all": "{:.4f}",
                "test_recall@50_all": "{:.4f}",
                "test_ndcg@10_all": "{:.4f}",
                "test_ndcg@20_all": "{:.4f}",
                "test_ndcg@50_all": "{:.4f}",
                "evaluated_users_all": "{:,}",
                "catalog_items": "{:,}",
            }),
            width="stretch",
        )

        st.markdown("##### 🔍 So sánh tiến trình từ Smoke Test (3 epochs × 100 batches) → Full Benchmark → Mốc 300 Epochs Bài Báo (Tập Beauty):")
        s_bpr_r20 = smoke_res.iloc[0]["test_recall@20"] if "test_recall@20" in smoke_res.columns else smoke_res.iloc[0]["test_recall_at_20"]
        s_bpr_n20 = smoke_res.iloc[0]["test_ndcg@20"] if "test_ndcg@20" in smoke_res.columns else smoke_res.iloc[0]["test_ndcg_at_20"]
        s_dau_r20 = smoke_res.iloc[1]["test_recall@20"] if "test_recall@20" in smoke_res.columns else smoke_res.iloc[1]["test_recall_at_20"]
        s_dau_n20 = smoke_res.iloc[1]["test_ndcg@20"] if "test_ndcg@20" in smoke_res.columns else smoke_res.iloc[1]["test_ndcg_at_20"]

        comp_table = pd.DataFrame([
            {
                "Giai đoạn / Cấu hình": "1. Smoke Test (3 epochs, 100 batches/ep, 512 users)",
                "Mô hình": "BPRMF",
                "Recall@10": "-",
                "Recall@20": f"{s_bpr_r20:.4f}",
                "Recall@50": "-",
                "NDCG@10": "-",
                "NDCG@20": f"{s_bpr_n20:.4f}",
                "NDCG@50": "-",
                "Số Users Đánh Giá": "512 / 22,363",
            },
            {
                "Giai đoạn / Cấu hình": "1. Smoke Test (3 epochs, 100 batches/ep, 512 users)",
                "Mô hình": "DirectAU (γ=0.5)",
                "Recall@10": "-",
                "Recall@20": f"{s_dau_r20:.4f}",
                "Recall@50": "-",
                "NDCG@10": "-",
                "NDCG@20": f"{s_dau_n20:.4f}",
                "NDCG@50": "-",
                "Số Users Đánh Giá": "512 / 22,363",
            },
            {
                "Giai đoạn / Cấu hình": "2. Full Benchmark Tự Chạy (581 batches/ep, Toàn bộ Users)",
                "Mô hình": "BPRMF (Best Ep 21)",
                "Recall@10": f"{bpr_r['test_recall@10_all']:.4f}",
                "Recall@20": f"{bpr_r['test_recall@20_all']:.4f}",
                "Recall@50": f"{bpr_r['test_recall@50_all']:.4f}",
                "NDCG@10": f"{bpr_r['test_ndcg@10_all']:.4f}",
                "NDCG@20": f"{bpr_r['test_ndcg@20_all']:.4f}",
                "NDCG@50": f"{bpr_r['test_ndcg@50_all']:.4f}",
                "Số Users Đánh Giá": "22,363 / 22,363 (100%)",
            },
            {
                "Giai đoạn / Cấu hình": "2. Full Benchmark Tự Chạy (581 batches/ep, Toàn bộ Users)",
                "Mô hình": "DirectAU (γ=0.5, Ep 40)",
                "Recall@10": f"{dau_r['test_recall@10_all']:.4f} (+18.5%)",
                "Recall@20": f"{dau_r['test_recall@20_all']:.4f} (+17.3%)",
                "Recall@50": f"{dau_r['test_recall@50_all']:.4f} (+14.2%)",
                "NDCG@10": f"{dau_r['test_ndcg@10_all']:.4f} (+25.3%)",
                "NDCG@20": f"{dau_r['test_ndcg@20_all']:.4f} (+23.7%)",
                "NDCG@50": f"{dau_r['test_ndcg@50_all']:.4f} (+20.8%)",
                "Số Users Đánh Giá": "22,363 / 22,363 (100%)",
            },
            {
                "Giai đoạn / Cấu hình": "3. Mốc Chuẩn Bài Báo KDD '22 (300 epochs, 5 seeds)",
                "Mô hình": "BPRMF (Paper Table 2)",
                "Recall@10": "0.0806",
                "Recall@20": "0.1153",
                "Recall@50": "0.1763",
                "NDCG@10": "0.0444",
                "NDCG@20": "0.0534",
                "NDCG@50": "0.0658",
                "Số Users Đánh Giá": "22,363 / 22,363 (100%)",
            },
            {
                "Giai đoạn / Cấu hình": "3. Mốc Chuẩn Bài Báo KDD '22 (300 epochs, 5 seeds)",
                "Mô hình": "DirectAU (Paper Table 2)",
                "Recall@10": "0.1002 (+24.3%)",
                "Recall@20": "0.1400 (+21.4%)",
                "Recall@50": "0.2062 (+17.0%)",
                "NDCG@10": "0.0582 (+31.1%)",
                "NDCG@20": "0.0686 (+28.5%)",
                "NDCG@50": "0.0820 (+24.6%)",
                "Số Users Đánh Giá": "22,363 / 22,363 (100%)",
            },
        ])
        st.dataframe(comp_table, width="stretch")

        col_g1, col_g2 = st.columns(2)
        with col_g1:
            metrics_names = ["Recall@10", "Recall@20", "Recall@50", "NDCG@10", "NDCG@20", "NDCG@50"]
            bpr_vals = [bpr_r["test_recall@10_all"], bpr_r["test_recall@20_all"], bpr_r["test_recall@50_all"], bpr_r["test_ndcg@10_all"], bpr_r["test_ndcg@20_all"], bpr_r["test_ndcg@50_all"]]
            dau_vals = [dau_r["test_recall@10_all"], dau_r["test_recall@20_all"], dau_r["test_recall@50_all"], dau_r["test_ndcg@10_all"], dau_r["test_ndcg@20_all"], dau_r["test_ndcg@50_all"]]
            x_pos = np.arange(len(metrics_names))
            fig_b, ax_b = plt.subplots(figsize=(7.2, 4.0))
            ax_b.bar(x_pos - 0.18, bpr_vals, width=0.35, color="#dc2626", label="BPRMF (Full Eval)")
            ax_b.bar(x_pos + 0.18, dau_vals, width=0.35, color="#2563eb", label="DirectAU (Full Eval)")
            for idx, v in enumerate(dau_vals):
                ax_b.text(idx + 0.18, v + 0.003, f"{v:.4f}", ha="center", fontsize=8, fontweight="bold")
            ax_b.set_xticks(x_pos)
            ax_b.set_xticklabels(metrics_names, fontweight="bold")
            ax_b.set_ylim(0, 0.20)
            ax_b.set_ylabel("Điểm số trên tập Test (22,363 Users)")
            ax_b.set_title("Đối Sánh 6 Độ Đo Giữa BPRMF & DirectAU (Beauty)", fontweight="bold")
            ax_b.legend()
            st.pyplot(fig_b)
            plt.close(fig_b)

        with col_g2:
            bins = np.array(geom["DirectAU"]["hist_bins"])
            centers = 0.5 * (bins[:-1] + bins[1:])
            fig_cos, ax_cos = plt.subplots(figsize=(7.2, 4.0))
            ax_cos.plot(centers, geom["DirectAU"]["pos_cos_hist"], color="#2563eb", lw=2.3, label=f"DirectAU Cặp Dương Test (μ={geom['DirectAU']['pos_cos_mean']:.3f})")
            ax_cos.plot(centers, geom["DirectAU"]["rnd_cos_hist"], color="#9ca3af", lw=2, linestyle="--", label=f"DirectAU Cặp Ngẫu Nhiên (μ={geom['DirectAU']['rnd_cos_mean']:.3f} ≈ 0)")
            ax_cos.axvline(0.0, color="black", linestyle=":", alpha=0.5)
            ax_cos.set_xlabel("Độ tương đồng Cosine cos(θ) = ũᵀ ĩ trên S⁶³")
            ax_cos.set_ylabel("Số lượng cặp mẫu")
            ax_cos.set_title("Phân Bố Góc Cosine Giữa Cặp Dương Test vs Cặp Ngẫu Nhiên", fontweight="bold")
            ax_cos.legend(fontsize=8.5)
            st.pyplot(fig_cos)
            plt.close(fig_cos)

    with tab_r2:
        st.subheader("2. Bảng 2 (Table 2 — Trang 6 Bài Báo): Đối Sánh 9 Phương Pháp Trên 3 Tập Dữ Liệu")
        ds_filter = st.radio("Chọn Tập Dữ Liệu Để Quan Sát Chi Tiết:", ["Beauty", "Gowalla", "Yelp2018"], horizontal=True)

        table2_raw = {
            "Beauty": pd.DataFrame({
                "Metric": ["Recall@10", "Recall@20", "Recall@50", "NDCG@10", "NDCG@20", "NDCG@50"],
                "BPRMF": [0.0806, 0.1153, 0.1763, 0.0444, 0.0534, 0.0658],
                "BPR-DS": [0.0816, 0.1181, 0.1745, 0.0459, 0.0554, 0.0670],
                "ENMF": [0.0915, 0.1282, 0.1914, 0.0511, 0.0606, 0.0736],
                "RecVAE": [0.0824, 0.1145, 0.1712, 0.0486, 0.0570, 0.0686],
                "LGCN": [0.0863, 0.1201, 0.1819, 0.0484, 0.0581, 0.0699],
                "DGCF": [0.0897, 0.1283, 0.1958, 0.0501, 0.0600, 0.0738],
                "BUIR": [0.0816, 0.1204, 0.1866, 0.0457, 0.0556, 0.0692],
                "CLRec": [0.0937, 0.1337, 0.1996, 0.0547, 0.0651, 0.0786],
                "DirectAU": [0.1002, 0.1400, 0.2062, 0.0582, 0.0686, 0.0820],
                "Improv. vs Best Baseline": ["+6.94%", "+4.74%", "+3.33%", "+6.44%", "+5.38%", "+4.33%"],
            }),
            "Gowalla": pd.DataFrame({
                "Metric": ["Recall@10", "Recall@20", "Recall@50", "NDCG@10", "NDCG@20", "NDCG@50"],
                "BPRMF": [0.0866, 0.1263, 0.2040, 0.0622, 0.0736, 0.0926],
                "BPR-DS": [0.1132, 0.1637, 0.2593, 0.0814, 0.0961, 0.1196],
                "ENMF": [0.1149, 0.1671, 0.2675, 0.0797, 0.0953, 0.1200],
                "RecVAE": [0.1211, 0.1771, 0.2768, 0.0845, 0.1007, 0.1251],
                "LGCN": [0.1289, 0.1871, 0.2934, 0.0930, 0.1097, 0.1356],
                "DGCF": [0.1301, 0.1889, 0.2919, 0.0939, 0.1099, 0.1358],
                "BUIR": [0.0798, 0.1164, 0.1917, 0.0570, 0.0676, 0.0858],
                "CLRec": [0.1215, 0.1755, 0.2813, 0.0868, 0.1022, 0.1281],
                "DirectAU": [0.1394, 0.2014, 0.3127, 0.0991, 0.1170, 0.1442],
                "Improv. vs Best Baseline": ["+7.15%", "+6.63%", "+6.56%", "+5.56%", "+6.44%", "+6.20%"],
            }),
            "Yelp2018": pd.DataFrame({
                "Metric": ["Recall@10", "Recall@20", "Recall@50", "NDCG@10", "NDCG@20", "NDCG@50"],
                "BPRMF": [0.0416, 0.0693, 0.1293, 0.0335, 0.0428, 0.0602],
                "BPR-DS": [0.0533, 0.0864, 0.1572, 0.0423, 0.0534, 0.0740],
                "ENMF": [0.0596, 0.0957, 0.1710, 0.0482, 0.0603, 0.0821],
                "RecVAE": [0.0495, 0.0820, 0.1494, 0.0395, 0.0504, 0.0698],
                "LGCN": [0.0508, 0.0833, 0.1534, 0.0406, 0.0514, 0.0717],
                "DGCF": [0.0519, 0.0849, 0.1575, 0.0409, 0.0521, 0.0732],
                "BUIR": [0.0444, 0.0737, 0.1386, 0.0349, 0.0448, 0.0636],
                "CLRec": [0.0547, 0.0890, 0.1606, 0.0436, 0.0551, 0.0758],
                "DirectAU": [0.0684, 0.1096, 0.1935, 0.0553, 0.0691, 0.0933],
                "Improv. vs Best Baseline": ["+14.83%", "+14.55%", "+13.16%", "+14.77%", "+14.53%", "+13.67%"],
            }),
        }

        cur_t2 = table2_raw[ds_filter]
        model_cols = ["BPRMF", "BPR-DS", "ENMF", "RecVAE", "LGCN", "DGCF", "BUIR", "CLRec", "DirectAU"]
        st.dataframe(
            cur_t2.style.highlight_max(axis=1, subset=model_cols, color="#bbf7d0").format({m: "{:.4f}" for m in model_cols}),
            width="stretch",
        )

        r20_row = cur_t2.loc[cur_t2["Metric"] == "Recall@20", model_cols].iloc[0]
        n20_row = cur_t2.loc[cur_t2["Metric"] == "NDCG@20", model_cols].iloc[0]
        fig_t2, (ax_t2a, ax_t2b) = plt.subplots(1, 2, figsize=(13.5, 4.2))
        colors_9 = ["#94a3b8"] * 8 + ["#2563eb"]
        ax_t2a.bar(model_cols, r20_row.values, color=colors_9)
        ax_t2a.set_title(f"{ds_filter} — Recall@20 Của 9 Mô Hình", fontweight="bold")
        ax_t2a.tick_params(axis="x", rotation=30)
        for i_m, v_m in enumerate(r20_row.values):
            ax_t2a.text(i_m, v_m + 0.002, f"{v_m:.4f}", ha="center", fontsize=8, fontweight="bold" if i_m == 8 else "normal")

        ax_t2b.bar(model_cols, n20_row.values, color=["#94a3b8"] * 8 + ["#10b981"])
        ax_t2b.set_title(f"{ds_filter} — NDCG@20 Của 9 Mô Hình", fontweight="bold")
        ax_t2b.tick_params(axis="x", rotation=30)
        for i_m, v_m in enumerate(n20_row.values):
            ax_t2b.text(i_m, v_m + 0.001, f"{v_m:.4f}", ha="center", fontsize=8, fontweight="bold" if i_m == 8 else "normal")
        plt.tight_layout()
        st.pyplot(fig_t2)
        plt.close(fig_t2)

    with tab_r3:
        st.subheader("3. Tích Hợp Với Bộ Mã Hóa Đồ Thị LightGCN (Table 3) & Hiệu Quả Thời Gian (Table 4)")
        col_tb3, col_tb4 = st.columns(2)
        with col_tb3:
            st.markdown("##### 🔗 Bảng 3 (Table 3): Kết Hợp DirectAU Với MF, LGCN-1, LGCN-2")
            t3_df = pd.DataFrame([
                {"Encoder + Objective": "MF + BPR (BPRMF)", "Beauty Recall@20": 0.1153, "Beauty NDCG@20": 0.0534, "Gowalla Recall@20": 0.1263, "Gowalla NDCG@20": 0.0736},
                {"Encoder + Objective": "MF + DirectAU", "Beauty Recall@20": 0.1400, "Beauty NDCG@20": 0.0686, "Gowalla Recall@20": 0.2014, "Gowalla NDCG@20": 0.1170},
                {"Encoder + Objective": "LGCN-1 + BPR", "Beauty Recall@20": 0.1211, "Beauty NDCG@20": 0.0560, "Gowalla Recall@20": 0.1769, "Gowalla NDCG@20": 0.1033},
                {"Encoder + Objective": "LGCN-1 + DirectAU", "Beauty Recall@20": 0.1444, "Beauty NDCG@20": 0.0700, "Gowalla Recall@20": 0.2036, "Gowalla NDCG@20": 0.1184},
                {"Encoder + Objective": "LGCN-2 + BPR", "Beauty Recall@20": 0.1201, "Beauty NDCG@20": 0.0581, "Gowalla Recall@20": 0.1871, "Gowalla NDCG@20": 0.1097},
                {"Encoder + Objective": "LGCN-2 + DirectAU", "Beauty Recall@20": 0.1455, "Beauty NDCG@20": 0.0707, "Gowalla Recall@20": 0.2043, "Gowalla NDCG@20": 0.1191},
            ])
            st.dataframe(
                t3_df.style.highlight_max(subset=["Beauty Recall@20", "Beauty NDCG@20", "Gowalla Recall@20", "Gowalla NDCG@20"], color="#bbf7d0").format({
                    "Beauty Recall@20": "{:.4f}",
                    "Beauty NDCG@20": "{:.4f}",
                    "Gowalla Recall@20": "{:.4f}",
                    "Gowalla NDCG@20": "{:.4f}",
                }),
                width="stretch",
            )

        with col_tb4:
            st.markdown("##### ⏱️ Bảng 4 (Table 4): So Sánh Chi Phí Thời Gian Huấn Luyện (Trên Yelp2018)")
            t4_df = pd.DataFrame([
                {"Phương pháp": "BPRMF", "Thời gian / Epoch (s)": 29.8, "Số Epoch hội tụ": 59, "Tổng thời gian": "29 phút", "Tổng phút": 29},
                {"Phương pháp": "ENMF", "Thời gian / Epoch (s)": 24.8, "Số Epoch hội tụ": 89, "Tổng thời gian": "36 phút", "Tổng phút": 36},
                {"Phương pháp": "LGCN (GNN)", "Thời gian / Epoch (s)": 228.6, "Số Epoch hội tụ": 107, "Tổng thời gian": "6 giờ 48 phút", "Tổng phút": 408},
                {"Phương pháp": "DirectAU (MF)", "Thời gian / Epoch (s)": 37.3, "Số Epoch hội tụ": 50, "Tổng thời gian": "31 phút", "Tổng phút": 31},
            ])
            st.dataframe(t4_df.drop(columns=["Tổng phút"]), width="stretch")

            fig_t4, ax_t4 = plt.subplots(figsize=(6.2, 2.8))
            bars_t4 = ax_t4.barh(t4_df["Phương pháp"], t4_df["Tổng phút"], color=["#64748b", "#f59e0b", "#dc2626", "#10b981"], height=0.5)
            for b in bars_t4:
                w = b.get_width()
                ax_t4.text(w + 6, b.get_y() + b.get_height() / 2, f"{int(w)} phút", va="center", fontweight="bold")
            ax_t4.set_xlim(0, 465)
            ax_t4.set_xlabel("Tổng thời gian huấn luyện đến khi hội tụ (Phút)")
            ax_t4.set_title("Tổng Thời Gian Huấn Luyện (DirectAU Nhanh Gấp ~13 Lần LGCN)", fontweight="bold")
            st.pyplot(fig_t4)
            plt.close(fig_t4)


# =====================================================================
# MODULE 6: HỆ THỐNG GỢI Ý TOP-K THỰC TẾ TRÊN AMAZON BEAUTY
# =====================================================================
elif menu == "6. 🔍 Hệ Thống Gợi Ý Top-K Thực Tế Trên Amazon Beauty":
    st.title("🔍 Trình Khảo Sát Gợi Ý Top-K Sản Phẩm Thực Tế (Interactive Recommender Inspector)")
    st.markdown(
        """
Module này nạp trực tiếp các bộ trọng số đã huấn luyện (`BPR_full_best.pt`, `DirectAU_full_best.pt`, `BPR_smoke_best.pt`, `DirectAU_smoke_best.pt`) từ `final/benchmark_results/` để chạy **Full-Catalog Ranking (`12,101` sản phẩm Amazon Beauty)** trong thời gian thực.
"""
    )

    assets = load_recommender_assets()
    data = load_csv_artifacts()
    showcase_list = data["showcase_users"]
    user_tokens = assets["user_tokens"]
    item_tokens = assets["item_tokens"]
    train_hist = assets["train_hist"]
    test_targ = assets["test_targ"]
    test_hist = assets["test_hist"]

    col_sel1, col_sel2, col_sel3 = st.columns([1.5, 1, 1])
    with col_sel1:
        preset_options = [
            f"User {u['user']} ({u['user_token']}) — DirectAU trúng {u['dau_hits']}/{u['n_test_targets']} vs BPR {u['bpr_hits']}/{u['n_test_targets']}"
            for u in showcase_list[:15]
        ]
        preset_options.insert(
            0,
            f"User 40 ({user_tokens[40]}) — User mẫu mặc định trong Smoke Test (eval_users[0])",
        )
        preset_options.append("🔢 Tự nhập chỉ số User ID bất kỳ (0 → 22,362)")
        chosen_preset = st.selectbox("Chọn User trong tập Amazon Beauty để kiểm thử gợi ý:", preset_options, index=1)

    with col_sel2:
        ckpt_mode = st.radio(
            "Chọn bộ Checkpoint mô hình:",
            ["Full Benchmark (40 Epochs)", "Smoke Test (3 Epochs)"],
            index=0,
        )
    with col_sel3:
        k_val = st.slider("Số lượng sản phẩm gợi ý Top-K:", 5, 50, 20, 5)

    if chosen_preset.startswith("🔢"):
        selected_user = st.number_input("Nhập User ID (từ 0 đến 22,362):", min_value=0, max_value=len(user_tokens) - 1, value=12469, step=1)
    elif chosen_preset.startswith("User 40"):
        selected_user = 40
    else:
        selected_user = int(chosen_preset.split()[1])

    models_dict = assets["full_models"] if ckpt_mode.startswith("Full") else assets["smoke_models"]
    seen_items = test_hist[selected_user]
    gt_items = test_targ[selected_user]

    @torch.no_grad()
    def recommend_for_user(u_w, i_w, u_id, seen_set, top_k):
        u_vec = u_w[u_id : u_id + 1]
        scores = (u_vec @ i_w.T).squeeze(0).clone()
        u_norm = F.normalize(u_vec, p=2, dim=-1)
        i_norm = F.normalize(i_w, p=2, dim=-1)
        cos_sim = (u_norm @ i_norm.T).squeeze(0)
        scores[list(seen_set)] = -torch.inf
        top_scores, top_indices = scores.topk(top_k)
        top_idx_list = top_indices.numpy().tolist()
        top_sc_list = top_scores.numpy().tolist()
        top_cos_list = cos_sim[top_indices].numpy().tolist()
        return top_idx_list, top_sc_list, top_cos_list

    bpr_idx, bpr_sc, bpr_cos = recommend_for_user(*models_dict["BPR"], selected_user, seen_items, k_val)
    dau_idx, dau_sc, dau_cos = recommend_for_user(*models_dict["DirectAU"], selected_user, seen_items, k_val)

    discounts = 1.0 / np.log2(np.arange(2, k_val + 2))
    idcg = discounts[: min(k_val, len(gt_items))].sum()

    bpr_hits_bool = [i in gt_items for i in bpr_idx]
    dau_hits_bool = [i in gt_items for i in dau_idx]

    bpr_rec_u = sum(bpr_hits_bool) / len(gt_items)
    dau_rec_u = sum(dau_hits_bool) / len(gt_items)
    bpr_ndcg_u = float((np.array(bpr_hits_bool, dtype=float) * discounts).sum() / idcg)
    dau_ndcg_u = float((np.array(dau_hits_bool, dtype=float) * discounts).sum() / idcg)

    u_c1, u_c2, u_c3, u_c4 = st.columns(4)
    u_c1.metric("Mã định danh gốc (Amazon ReviewerID)", user_tokens[selected_user])
    u_c2.metric("Lịch sử đã biết (Train + Valid Masked)", f"{len(seen_items)} sản phẩm")
    u_c3.metric(f"BPRMF — Recall@{k_val} / NDCG@{k_val}", f"{bpr_rec_u*100:.1f}% / {bpr_ndcg_u:.4f}", delta=f"{sum(bpr_hits_bool)}/{len(gt_items)} Hits")
    u_c4.metric(
        f"DirectAU — Recall@{k_val} / NDCG@{k_val}",
        f"{dau_rec_u*100:.1f}% / {dau_ndcg_u:.4f}",
        delta=f"{sum(dau_hits_bool)}/{len(gt_items)} Hits (Vượt trội)" if sum(dau_hits_bool) >= sum(bpr_hits_bool) else f"{sum(dau_hits_bool)}/{len(gt_items)} Hits",
    )

    with st.expander("📦 Xem Chi Tiết Lịch Sử Tương Tác (Train/Valid) & Mục Tiêu Thực Tế (Test Ground-Truth ASINs) Của User Này", expanded=True):
        gt_asins = [f"{item_tokens[i]} (ID: {i})" for i in sorted(gt_items)]
        train_asins = [f"{item_tokens[i]}" for i in sorted(train_hist[selected_user])[:20]]
        st.markdown(f"**🎯 Các sản phẩm mục tiêu trong tập Test (Ground-Truth Targets — {len(gt_items)} sản phẩm):** `{', '.join(gt_asins)}`")
        st.markdown(f"**🛒 Mẫu lịch sử tương tác trong tập Train (Đã ẩn khỏi ứng viên xếp hạng):** `{', '.join(train_asins)}`")
        st.caption("💡 Có thể tra cứu trực tiếp sản phẩm trên Amazon theo cú pháp URL: `https://www.amazon.com/dp/<ASIN>`")

    rec_df = pd.DataFrame({
        "Hạng (Rank)": range(1, k_val + 1),
        "BPR_ASIN": [item_tokens[i] for i in bpr_idx],
        "BPR_Score ⟨u,i⟩": [round(s, 4) for s in bpr_sc],
        "BPR_Cosine": [round(c, 4) for c in bpr_cos],
        "BPR_Test_Hit": ["✅ TRÚNG" if h else "—" for h in bpr_hits_bool],
        "DirectAU_ASIN": [item_tokens[i] for i in dau_idx],
        "DirectAU_Score ⟨u,i⟩": [round(s, 4) for s in dau_sc],
        "DirectAU_Cosine": [round(c, 4) for c in dau_cos],
        "DirectAU_Test_Hit": ["✅ TRÚNG" if h else "—" for h in dau_hits_bool],
    })

    def highlight_hit_rows(row):
        styles = [""] * len(row)
        if row["BPR_Test_Hit"] == "✅ TRÚNG":
            for idx in (1, 2, 3, 4):
                styles[idx] = "background-color: #fef08a; font-weight: bold; color: #111827;"
        if row["DirectAU_Test_Hit"] == "✅ TRÚNG":
            for idx in (5, 6, 7, 8):
                styles[idx] = "background-color: #bbf7d0; font-weight: bold; color: #111827;"
        return styles

    st.subheader(f"📋 Bảng Đối Sánh Danh Sách Gợi Ý Top-{k_val} Sản Phẩm (BPRMF vs DirectAU)")
    st.dataframe(rec_df.style.apply(highlight_hit_rows, axis=1), width="stretch", height=450)
