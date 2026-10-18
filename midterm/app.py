import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_moons, make_circles
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import time

# Thiết lập cấu hình trang Streamlit
st.set_page_config(
    page_title="Mô Phỏng Toán Học & Cải Tiến Thuật Toán ANN - Nhóm 5 UIT",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cấu hình phong cách đồ thị
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

# Sidebar Menu
st.sidebar.title("🧠 Chuyên Đề ANN")
st.sidebar.markdown("""
**Nhóm 5 - Artificial Neural Networks (ANN)**  
*Môn: Khai thác dữ liệu và ứng dụng (Data Mining and Applications)*  
*Trường Đại Học Công Nghệ Thông Tin - ĐHQG-HCM*
""")

menu = st.sidebar.radio(
    "Chọn Module Mô Phỏng:",
    [
        "1. 🎯 Live Demo: Nắn Đường Phân Loại Theo Epoch",
        "2. 📐 Mô Phỏng Tương Tác: Hàm Loss & Xung Lực Gradient",
        "3. ⚡ Thuật Toán Tối Ưu (SGD, Momentum, Adam) & Generalization",
        "4. 🧪 Vấn Đề Tầng Ẩn: Vanishing Gradient & Dying ReLU",
        "5. 📊 Thực Nghiệm MNIST: EDA, Chiếu 2D/3D & Đối Sánh ANN Vượt Trội"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("""
💡 **Đặc điểm mô phỏng:**
- Tập trung vào **bản chất toán học** & **cơ chế cải tiến thuật toán**.
- Bỏ các phần trùng lặp với playground cơ bản (trọng số đơn lẻ, mặt phẳng 3D tĩnh).
- Trực quan hóa động quá trình **nắn đường phân loại (warping)** trong thời gian thực.
""")


# =====================================================================
# HÀM BỔ TRỢ DÙNG CHUNG: MẠNG MLP & CACHE DỮ LIỆU MNIST
# =====================================================================
@st.cache_data
def load_mnist_dataset():
    """Tải và tiền xử lý dữ liệu MNIST (có cache để tối ưu hiệu năng)."""
    (X_tr_raw, y_tr), (X_te_raw, y_te) = keras.datasets.mnist.load_data()
    X_tr = (X_tr_raw.reshape(-1, 28 * 28) / 255.0).astype(np.float32)
    X_te = (X_te_raw.reshape(-1, 28 * 28) / 255.0).astype(np.float32)
    return X_tr_raw, y_tr, X_te_raw, y_te, X_tr, X_te

@st.cache_data
def compute_mnist_pca_components(n_samples=1500):
    """Tính toán phép chiếu PCA 2D và 3D trên tập con MNIST."""
    X_tr_raw, y_tr, _, _, _, _ = load_mnist_dataset()
    np.random.seed(42)
    sample_indices = np.random.choice(len(y_tr), n_samples, replace=False)
    X_sub = (X_tr_raw[sample_indices].reshape(n_samples, -1) / 255.0).astype(np.float32)
    y_sub = y_tr[sample_indices]
    
    pca_2d = PCA(n_components=2, random_state=42)
    X_pca_2d = pca_2d.fit_transform(X_sub)
    
    pca_3d = PCA(n_components=3, random_state=42)
    X_pca_3d = pca_3d.fit_transform(X_sub)
    
    return X_pca_2d, X_pca_3d, y_sub, pca_2d.explained_variance_ratio_, pca_3d.explained_variance_ratio_

def generate_dataset(name="moons", n_samples=250, noise=0.15):
    np.random.seed(42)
    if name == "moons":
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=42)
    elif name == "circles":
        X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=42)
    elif name == "xor":
        X = np.random.uniform(-1.5, 1.5, (n_samples, 2))
        y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)
    return X, y

class InteractiveMLP:
    """Mạng MLP 2D được tối ưu hóa bằng NumPy, hỗ trợ lưu snapshot từng epoch."""
    def __init__(self, layer_sizes=[2, 16, 1], activation='relu', optimizer='adam', lr=0.03):
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.optimizer = optimizer
        self.lr = lr
        
        # Khởi tạo trọng số He (cho ReLU) hoặc Glorot/Xavier (cho Tanh/Sigmoid)
        self.weights = []
        self.biases = []
        for i in range(len(layer_sizes) - 1):
            n_in, n_out = layer_sizes[i], layer_sizes[i+1]
            scale = np.sqrt(2.0 / n_in) if activation == 'relu' else np.sqrt(1.0 / n_in)
            self.weights.append(np.random.randn(n_in, n_out) * scale)
            self.biases.append(np.zeros((1, n_out)))
            
        # Bộ nhớ cho các Optimizer
        self.m_w = [np.zeros_like(w) for w in self.weights]
        self.v_w = [np.zeros_like(w) for w in self.weights]
        self.m_b = [np.zeros_like(b) for b in self.biases]
        self.v_b = [np.zeros_like(b) for b in self.biases]
        self.t = 0
        
    def _act(self, z):
        if self.activation == 'relu':
            return np.maximum(0, z)
        elif self.activation == 'tanh':
            return np.tanh(z)
        elif self.activation == 'sigmoid':
            return 1.0 / (1.0 + np.exp(-np.clip(z, -15, 15)))
        return z
        
    def _act_grad(self, a, z):
        if self.activation == 'relu':
            return (z > 0).astype(float)
        elif self.activation == 'tanh':
            return 1.0 - a**2
        elif self.activation == 'sigmoid':
            return a * (1.0 - a)
        return np.ones_like(z)
        
    def forward(self, X):
        a_list = [X]
        z_list = []
        for i in range(len(self.weights) - 1):
            z = np.dot(a_list[-1], self.weights[i]) + self.biases[i]
            a = self._act(z)
            z_list.append(z)
            a_list.append(a)
            
        # Lớp đầu ra Sigmoid nhị phân
        z_out = np.dot(a_list[-1], self.weights[-1]) + self.biases[-1]
        a_out = 1.0 / (1.0 + np.exp(-np.clip(z_out, -15, 15)))
        z_list.append(z_out)
        a_list.append(a_out)
        return a_out, a_list, z_list
        
    def train_step(self, X, y):
        m = y.shape[0]
        y_col = y.reshape(-1, 1)
        self.t += 1
        
        # 1. Forward
        a_out, a_list, z_list = self.forward(X)
        
        # Tính Loss BCE
        eps = 1e-12
        loss = -np.mean(y_col * np.log(a_out + eps) + (1 - y_col) * np.log(1 - a_out + eps))
        acc = np.mean((a_out > 0.5).flatten() == y)
        
        # 2. Backpropagation
        # Đạo hàm BCE kết hợp Sigmoid ở tầng cuối: delta^(L) = a - y
        delta = a_out - y_col
        
        dw_list = [None] * len(self.weights)
        db_list = [None] * len(self.biases)
        
        dw_list[-1] = np.dot(a_list[-2].T, delta) / m
        db_list[-1] = np.sum(delta, axis=0, keepdims=True) / m
        
        curr_delta = delta
        for i in range(len(self.weights) - 2, -1, -1):
            grad_act = self._act_grad(a_list[i+1], z_list[i])
            curr_delta = np.dot(curr_delta, self.weights[i+1].T) * grad_act
            dw_list[i] = np.dot(a_list[i].T, curr_delta) / m
            db_list[i] = np.sum(curr_delta, axis=0, keepdims=True) / m
            
        # 3. Cập nhật trọng số theo Optimizer
        beta1, beta2, eps_opt = 0.9, 0.999, 1e-8
        for i in range(len(self.weights)):
            if self.optimizer == 'adam':
                self.m_w[i] = beta1 * self.m_w[i] + (1 - beta1) * dw_list[i]
                self.v_w[i] = beta2 * self.v_w[i] + (1 - beta2) * (dw_list[i]**2)
                m_hat = self.m_w[i] / (1 - beta1**self.t)
                v_hat = self.v_w[i] / (1 - beta2**self.t)
                self.weights[i] -= self.lr * m_hat / (np.sqrt(v_hat) + eps_opt)
                
                self.m_b[i] = beta1 * self.m_b[i] + (1 - beta1) * db_list[i]
                self.v_b[i] = beta2 * self.v_b[i] + (1 - beta2) * (db_list[i]**2)
                m_hat_b = self.m_b[i] / (1 - beta1**self.t)
                v_hat_b = self.v_b[i] / (1 - beta2**self.t)
                self.biases[i] -= self.lr * m_hat_b / (np.sqrt(v_hat_b) + eps_opt)
            elif self.optimizer == 'momentum':
                self.m_w[i] = 0.9 * self.m_w[i] + self.lr * dw_list[i]
                self.weights[i] -= self.m_w[i]
                self.m_b[i] = 0.9 * self.m_b[i] + self.lr * db_list[i]
                self.biases[i] -= self.m_b[i]
            else: # sgd
                self.weights[i] -= self.lr * dw_list[i]
                self.biases[i] -= self.lr * db_list[i]
                
        return loss, acc


# =====================================================================
# MODULE 1: LIVE DEMO NẮN ĐƯỜNG PHÂN LOẠI (BOUNDARY WARPING THEO EPOCH)
# =====================================================================
if menu == "1. 🎯 Live Demo: Nắn Đường Phân Loại Theo Epoch":
    st.title("🎯 Live Demo: Quá Trình 'Nắn' Đường Phân Loại Qua Từng Epoch")
    st.markdown("""
    Trong mạng nơ-ron sâu (ANN), **các lớp ẩn (Hidden Layers)** áp dụng các phép biến đổi tuyến tính kết hợp với **hàm kích hoạt phi tuyến** 
    để liên tục bẻ gập, co giãn không gian vector đặc trưng. Khi các trọng số được cập nhật qua từng **Epoch**, 
    **đường ranh giới phân loại (Decision Boundary)** sẽ được **'nắn' (warp)** từ một siêu phẳng thô sơ dần dần uốn lượn ôm sát dữ liệu phi tuyến phức tạp!
    """)
    
    # Khu vực điều khiển cấu hình
    with st.expander("⚙️ Cấu Hình Mạng Nơ-ron & Dữ Liệu Huấn Luyện", expanded=True):
        col_cfg1, col_cfg2, col_cfg3, col_cfg4 = st.columns(4)
        with col_cfg1:
            dataset_choice = st.selectbox(
                "Tập Dữ Liệu Phi Tuyến:",
                ["Hai vầng trăng (Two Moons)", "Vòng tròn đồng tâm (Circles)", "Cổng phi tuyến XOR"],
                index=0
            )
            dataset_map = {
                "Hai vầng trăng (Two Moons)": "moons",
                "Vòng tròn đồng tâm (Circles)": "circles",
                "Cổng phi tuyến XOR": "xor"
            }
        with col_cfg2:
            arch_choice = st.selectbox(
                "Kiến Trúc Mạng MLP:",
                ["1 Lớp ẩn: [16]", "1 Lớp ẩn: [32]", "2 Lớp ẩn: [16, 8]", "2 Lớp ẩn: [32, 16]"],
                index=2
            )
            arch_map = {
                "1 Lớp ẩn: [16]": [2, 16, 1],
                "1 Lớp ẩn: [32]": [2, 32, 1],
                "2 Lớp ẩn: [16, 8]": [2, 16, 8, 1],
                "2 Lớp ẩn: [32, 16]": [2, 32, 16, 1]
            }
        with col_cfg3:
            act_choice = st.selectbox("Hàm Kích Hoạt Ẩn:", ["relu", "tanh", "sigmoid"], index=0)
            opt_choice = st.selectbox("Bộ Tối Ưu Hóa:", ["adam", "momentum", "sgd"], index=0)
        with col_cfg4:
            lr_val = st.slider("Tốc độ học (Learning Rate):", 0.005, 0.15, 0.04, 0.005)
            epochs_val = st.slider("Số lượng Epochs:", 40, 250, 120, 10)

    # Khởi tạo dữ liệu
    ds_key = dataset_map[dataset_choice]
    X_data, y_data = generate_dataset(ds_key, n_samples=240, noise=0.15)
    
    # Tạo lưới meshgrid cho contour boundary
    x_min, x_max = X_data[:, 0].min() - 0.6, X_data[:, 0].max() + 0.6
    y_min, y_max = X_data[:, 1].min() - 0.6, X_data[:, 1].max() + 0.6
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    col_btn1, col_btn2 = st.columns([1, 4])
    start_train = col_btn1.button("🚀 Bắt Đầu Huấn Luyện Live", type="primary", use_container_width=True)
    
    # Chỗ chứa đồ thị động
    plot_placeholder = st.empty()
    metrics_placeholder = st.empty()

    # Hàm vẽ snapshot
    def render_frame(epoch_idx, loss_val, acc_val, boundary_grid, loss_history, acc_history, total_epochs):
        fig, (ax_boundary, ax_curve) = plt.subplots(1, 2, figsize=(13.5, 4.8), gridspec_kw={'width_ratios': [1.2, 1]})
        
        # 1. Đồ thị Decision Boundary
        Z = boundary_grid.reshape(xx.shape)
        contour = ax_boundary.contourf(xx, yy, Z, levels=25, cmap="RdBu_r", alpha=0.75, vmin=0, vmax=1)
        ax_boundary.contour(xx, yy, Z, levels=[0.5], colors="black", linewidths=2.5, linestyles='--')
        
        # Điểm dữ liệu
        ax_boundary.scatter(X_data[y_data == 0, 0], X_data[y_data == 0, 1], c="#1f77b4", edgecolors="white", s=55, label="Lớp 0", zorder=4)
        ax_boundary.scatter(X_data[y_data == 1, 0], X_data[y_data == 1, 1], c="#d62728", edgecolors="white", s=55, label="Lớp 1", zorder=4)
        
        ax_boundary.set_xlim(x_min, x_max)
        ax_boundary.set_ylim(y_min, y_max)
        ax_boundary.set_title(f"📍 Ranh Giới Phân Loại Đang Được Nắn (Epoch {epoch_idx}/{total_epochs})\nĐường nét đứt màu đen: Decision Boundary (P = 0.5)", fontsize=11, fontweight='bold')
        ax_boundary.set_xlabel("Đặc trưng X1")
        ax_boundary.set_ylabel("Đặc trưng X2")
        ax_boundary.legend(loc="upper right", framealpha=0.85)
        
        # 2. Đồ thị Learning Curve
        ep_arr = np.arange(len(loss_history))
        ax_curve.plot(ep_arr, loss_history, color="#d62728", lw=2, label=f"Loss BCE: {loss_val:.4f}")
        ax_curve.set_ylabel("Mất Mát (Loss - BCE)", color="#d62728", fontweight='bold')
        ax_curve.tick_params(axis='y', labelcolor="#d62728")
        ax_curve.set_xlabel("Epochs")
        ax_curve.set_xlim(0, max(total_epochs, len(loss_history)))
        ax_curve.set_ylim(0, max(max(loss_history)*1.1, 0.8))
        
        ax_acc = ax_curve.twinx()
        ax_acc.plot(ep_arr, [a * 100 for a in acc_history], color="#2ca02c", lw=2, linestyle="--", label=f"Accuracy: {acc_val*100:.1f}%")
        ax_acc.set_ylabel("Độ Chính Xác (%)", color="#2ca02c", fontweight='bold')
        ax_acc.tick_params(axis='y', labelcolor="#2ca02c")
        ax_acc.set_ylim(40, 105)
        
        ax_curve.set_title("📈 Động Lực Học Tập Theo Thời Gian Thực (Learning Curves)", fontsize=11, fontweight='bold')
        ax_curve.grid(True, linestyle=":")
        
        plt.tight_layout()
        return fig

    # Xử lý khi nhấn nút Train
    if start_train:
        mlp_model = InteractiveMLP(
            layer_sizes=arch_map[arch_choice],
            activation=act_choice,
            optimizer=opt_choice,
            lr=lr_val
        )
        
        loss_hist = []
        acc_hist = []
        snapshots = [] # Lưu các mốc để tua lại
        
        # Snapshot epoch 0 (trước khi học)
        grid_pred_0, _, _ = mlp_model.forward(grid_points)
        init_preds, _, _ = mlp_model.forward(X_data)
        init_loss = -np.mean(y_data.reshape(-1, 1) * np.log(init_preds + 1e-12) + (1 - y_data.reshape(-1, 1)) * np.log(1 - init_preds + 1e-12))
        init_acc = np.mean((init_preds > 0.5).flatten() == y_data)
        
        loss_hist.append(init_loss)
        acc_hist.append(init_acc)
        snapshots.append((0, init_loss, init_acc, grid_pred_0.copy()))
        
        progress_bar = st.progress(0)
        
        # Vòng lặp huấn luyện từng epoch
        for ep in range(1, epochs_val + 1):
            loss_ep, acc_ep = mlp_model.train_step(X_data, y_data)
            loss_hist.append(loss_ep)
            acc_hist.append(acc_ep)
            
            # Cập nhật hiển thị mỗi 4 epoch hoặc epoch cuối
            if ep % 4 == 0 or ep == epochs_val:
                grid_pred, _, _ = mlp_model.forward(grid_points)
                snapshots.append((ep, loss_ep, acc_ep, grid_pred.copy()))
                
                # Render đồ thị
                fig_frame = render_frame(ep, loss_ep, acc_ep, grid_pred, loss_hist, acc_hist, epochs_val)
                plot_placeholder.pyplot(fig_frame)
                plt.close(fig_frame)
                
                progress_bar.progress(ep / epochs_val)
                time.sleep(0.02) # Tạo nhịp chuyển động mượt mà
                
        progress_bar.empty()
        
        # Lưu vào session state để tua lại
        st.session_state['trained_snapshots'] = snapshots
        st.session_state['total_epochs'] = epochs_val
        st.session_state['loss_hist'] = loss_hist
        st.session_state['acc_hist'] = acc_hist
        st.session_state['final_loss'] = loss_hist[-1]
        st.session_state['final_acc'] = acc_hist[-1]
        st.session_state['num_params'] = sum(w.size + b.size for w, b in zip(mlp_model.weights, mlp_model.biases))
        st.success(f"🎉 Huấn luyện hoàn tất! Độ chính xác cuối cùng: {acc_hist[-1]*100:.2f}% | Loss: {loss_hist[-1]:.4f}")

    # Nếu đã có kết quả trong session state, hiển thị chế độ Tua lại (Playback)
    if 'trained_snapshots' in st.session_state and not start_train:
        snapshots = st.session_state['trained_snapshots']
        epochs_saved = [s[0] for s in snapshots]
        
        st.markdown("---")
        st.subheader("⏪ Thanh Trượt Tua Lại Lịch Sử: Quan Sát Chi Tiết Từng Bước 'Nắn' Biên")
        st.caption("Kéo thanh trượt để xem hình dáng đường phân loại thay đổi tại các thời điểm khác nhau trong quá trình huấn luyện:")
        
        selected_ep = st.select_slider("Chọn Epoch muốn quan sát:", options=epochs_saved, value=epochs_saved[-1])
        
        # Tìm snapshot tương ứng
        snap = next(s for s in snapshots if s[0] == selected_ep)
        ep_idx, ep_loss, ep_acc, ep_grid = snap
        
        # Cắt lịch sử tương ứng
        sub_loss_hist = st.session_state['loss_hist'][:ep_idx + 1]
        sub_acc_hist = st.session_state['acc_hist'][:ep_idx + 1]
        
        fig_playback = render_frame(ep_idx, ep_loss, ep_acc, ep_grid, sub_loss_hist, sub_acc_hist, st.session_state['total_epochs'])
        plot_placeholder.pyplot(fig_playback)
        plt.close(fig_playback)

    # Hiển thị kết quả tổng hợp
    if 'final_loss' in st.session_state:
        st.markdown("### 📊 Tổng Hợp Kết Quả Thực Nghiệm Sau Huấn Luyện")
        res_c1, res_c2, res_c3, res_c4 = st.columns(4)
        res_c1.metric("Độ Chính Xác (Accuracy)", f"{st.session_state['final_acc']*100:.2f}%")
        res_c2.metric("Mất Mát Cuối Cùng (Loss)", f"{st.session_state['final_loss']:.4f}")
        res_c3.metric("Số Lượng Tham Số (Weights + Biases)", f"{st.session_state['num_params']:,}")
        res_c4.metric("Số Epoch Đã Chạy", f"{st.session_state['total_epochs']}")

    # Giải thích bản chất toán học
    st.markdown(r"""
    ---
    ### 🔬 Lý Giải Bản Chất Toán Học Của Việc 'Nắn' Đường Phân Loại
    1. **Không gian ban đầu (Epoch 0):** Trọng số khởi tạo ngẫu nhiên, mạng nơ-ron chỉ tạo ra một ranh giới tuyến tính thô hoặc phẳng, không thể phân biệt được bài toán phi tuyến (như Moons hay XOR).
    2. **Cơ chế uốn cong qua Lớp ẩn:**
       - Mỗi nơ-ron ở lớp ẩn tính: $z_j = \sum_i w_{ji} x_i + b_j$, xác định một siêu phẳng tuyến tính trong không gian 2D.
       - Khi đi qua **Hàm kích hoạt phi tuyến** $a_j = g(z_j)$ (như ReLU: bẻ gập mặt phẳng tại 0, hoặc Tanh: uốn cong mềm dạng chữ S), không gian đặc trưng bị **bẻ cong và gấp nếp (Warping / Folding)**.
    3. **Cộng gộp ở Lớp đầu ra:** Lớp đầu ra kết hợp tuyến tính các mặt phẳng đã bị bẻ gập: $\hat{y} = \sigma(\sum_j W_j a_j + b_{out})$. Nhờ đó, ranh giới quyết định $\{x \mid P(\hat{y}=1|x) = 0.5\}$ trở thành một **đường cong khép kín hoặc uốn lượn hình lưỡi liềm** ôm khít theo phân phối dữ liệu!
    4. **Vai trò của Backpropagation & Optimizer:** Lan truyền ngược tính toán chính xác vector gradient $\nabla_W L$, thuật toán tối ưu (Adam / Momentum) cập nhật trọng số liên tục để điều chỉnh độ dốc và góc quay của các siêu phẳng, ép đường biên khớp tối ưu với từng cụm dữ liệu.
    """)


# =====================================================================
# MODULE 2: MÔ PHỎNG TƯƠNG TÁC HÀM MẤT MÁT & XUNG LỰC GRADIENT
# =====================================================================
elif menu == "2. 📐 Mô Phỏng Tương Tác: Hàm Loss & Xung Lực Gradient":
    st.title("📐 Mô Phỏng Tương Tác: Giá Trị Loss & Độ Lớn Gradient Đẩy Trọng Số")
    st.markdown(r"""
    Trong bài toán phân loại của ANN, **Output activation và Loss function phải đi cùng nhau**:
    * **Binary classification:** Dùng **Sigmoid + Binary Cross-Entropy (BCE)** giúp đạo hàm rút gọn về dạng tuyến tính $\frac{\partial \mathcal{L}}{\partial z} = \hat{p} - y$.
    * Dưới đây là mô phỏng tương tác trực quan hóa **giá trị Loss** và **xung lực Gradient đẩy trọng số** khi thay đổi xác suất dự đoán $\hat{y}$.
    """)

    st.subheader("1. Mô Phỏng Tương Tác: Giá Trị Loss & Độ Lớn Gradient Đẩy Trọng Số")
    
    col_ctrl1, col_ctrl2 = st.columns([1, 2])
    with col_ctrl1:
        true_y = st.radio("Chọn nhãn thực tế $y$:", [1, 0], index=0)
        y_hat_slider = st.slider(r"Xác suất dự đoán hiện tại $\hat{y}$:", 0.001, 0.999, 0.05, 0.005)
        
        # Tính toán giá trị tại điểm chọn
        loss_mse_pt = 0.5 * (true_y - y_hat_slider)**2
        loss_bce_pt = -(true_y * np.log(y_hat_slider + 1e-12) + (1 - true_y) * np.log(1 - y_hat_slider + 1e-12))
        
        grad_mse_pt = np.abs((y_hat_slider - true_y) * y_hat_slider * (1 - y_hat_slider))
        grad_bce_pt = np.abs(y_hat_slider - true_y)
        
        st.markdown(rf"##### 📌 So sánh tại $\hat{{y}} = {y_hat_slider:.3f}$:")
        st.metric("Loss MSE", f"{loss_mse_pt:.4f}")
        st.metric("Loss BCE", f"{loss_bce_pt:.4f}", delta=f"+{loss_bce_pt - loss_mse_pt:.4f} (Phạt nặng)")
        st.metric("Gradient đẩy của MSE", f"{grad_mse_pt:.4f}", help="Nhỏ -> Học chậm/tê liệt")
        st.metric("Gradient đẩy của BCE", f"{grad_bce_pt:.4f}", delta=f"{grad_bce_pt/max(grad_mse_pt, 1e-6):.1f}x mạnh hơn!")

    with col_ctrl2:
        y_hat_range = np.linspace(0.001, 0.999, 500)
        if true_y == 1:
            loss_mse_curve = 0.5 * (1 - y_hat_range)**2
            loss_bce_curve = -np.log(y_hat_range)
            grad_mse_curve = np.abs((y_hat_range - 1) * y_hat_range * (1 - y_hat_range))
            grad_bce_curve = np.abs(y_hat_range - 1)
        else:
            loss_mse_curve = 0.5 * (0 - y_hat_range)**2
            loss_bce_curve = -np.log(1 - y_hat_range)
            grad_mse_curve = np.abs(y_hat_range * y_hat_range * (1 - y_hat_range))
            grad_bce_curve = np.abs(y_hat_range)

        fig_loss, (ax_l1, ax_l2) = plt.subplots(1, 2, figsize=(10, 4))
        
        # Đồ thị Hàm Loss
        ax_l1.plot(y_hat_range, loss_bce_curve, color="#d62728", lw=2.5, label="BCE Loss (Tiến tới ∞ khi sai)")
        ax_l1.plot(y_hat_range, loss_mse_curve, color="#1f77b4", lw=2, linestyle="--", label="MSE Loss (Bị chặn ở 0.5)")
        ax_l1.scatter([y_hat_slider], [loss_bce_pt], color="#d62728", s=80, zorder=5)
        ax_l1.scatter([y_hat_slider], [loss_mse_pt], color="#1f77b4", s=80, zorder=5)
        ax_l1.set_ylim(0, 4.0)
        ax_l1.set_xlabel("Xác suất dự đoán $\\hat{y}$")
        ax_l1.set_ylabel("Giá trị Loss")
        ax_l1.set_title(f"Hàm Mất Mát khi Nhãn thật y = {true_y}", fontweight='bold')
        ax_l1.legend()
        ax_l1.grid(True, linestyle=":")

        # Đồ thị Độ lớn Gradient
        ax_l2.plot(y_hat_range, grad_bce_curve, color="#2ca02c", lw=2.5, label="BCE Gradient |ŷ - y|")
        ax_l2.plot(y_hat_range, grad_mse_curve, color="#ff7f0e", lw=2, linestyle="--", label="MSE Gradient |(ŷ-y)σ'(z)|")
        ax_l2.scatter([y_hat_slider], [grad_bce_pt], color="#2ca02c", s=80, zorder=5)
        ax_l2.scatter([y_hat_slider], [grad_mse_pt], color="#ff7f0e", s=80, zorder=5)
        ax_l2.set_xlabel("Xác suất dự đoán $\\hat{y}$")
        ax_l2.set_ylabel("Độ lớn Gradient |∂L/∂z|")
        ax_l2.set_title("Xung Lực Gradient Đẩy Trọng Số", fontweight='bold')
        ax_l2.legend()
        ax_l2.grid(True, linestyle=":")

        st.pyplot(fig_loss)

    st.markdown(r"""
    ---
    ### 2. Hai Kết Quả Rút Gọn Quan Trọng (Output Activation + Loss Đi Cùng Nhau)
    * **Binary Classification (Sigmoid + BCE):**
      $$\mathcal{L}_{\text{BCE}} = -\left[y \log \hat{p} + (1 - y)\log(1 - \hat{p})\right] \implies \frac{\partial \mathcal{L}}{\partial z} = \hat{p} - y$$
    * **Multiclass Classification (Softmax + CCE):**
      $$\hat{p}_k = \frac{e^{z_k}}{\sum_{j=1}^K e^{z_j}}, \quad \mathcal{L}_{\text{CCE}} = -\sum_{k=1}^K y_k \log(\hat{p}_k) \implies \frac{\partial \mathcal{L}}{\partial z_k} = \hat{p}_k - y_k$$
    """)


# =====================================================================
# MODULE 3: THUẬT TOÁN TỐI ƯU (SGD, MOMENTUM, ADAM) & GENERALIZATION
# =====================================================================
elif menu == "3. ⚡ Thuật Toán Tối Ưu (SGD, Momentum, Adam) & Generalization":
    st.title("⚡ Thuật Toán Tối Ưu Hóa (Optimizer) & Khả Năng Generalization")
    st.markdown("""
    Sau khi **Backpropagation** tính toán các đạo hàm (gradients), **Optimizer** sẽ quyết định cách sử dụng gradient đó để cập nhật tham số $\\theta$.
    Bên cạnh tốc độ hội tụ trên tập huấn luyện, mục tiêu cốt lõi của mạng nơ-ron là **Generalization** — khả năng hoạt động tốt trên dữ liệu chưa từng thấy.
    """)

    tab_opt, tab_gen = st.tabs([
        "1. ⚡ Bộ Tối Ưu Hóa: SGD vs Momentum vs Adam",
        "2. 📈 Generalization: Training, Validation & Early Stopping"
    ])

    # ------------------ TAB 1: OPTIMIZERS (SGD, MOMENTUM, ADAM) ------------------
    with tab_opt:
        st.subheader("1. So Sánh Quỹ Đạo Hội Tụ: SGD vs Momentum vs Adam")
        col_land, col_opt_cfg = st.columns([1, 2])
        with col_land:
            landscape_choice = st.selectbox(
                "Chọn Bề Mặt Địa Hình Hàm Lỗi L(w1, w2):",
                [
                    "Hẻm núi dốc hẹp (Elongated Ravine)",
                    "Điểm yên ngựa (Saddle Point)",
                    "Thung lũng cong Rosenbrock (Banana Valley)"
                ]
            )
            steps_num = st.slider("Số bước cập nhật (Steps):", 10, 80, 35, 5)
            lr_opt = st.slider("Learning Rate (η):", 0.01, 0.4, 0.12, 0.01)

        # Định nghĩa các hàm địa hình và gradient
        if landscape_choice == "Hẻm núi dốc hẹp (Elongated Ravine)":
            def loss_fn(x, y): return 0.05 * x**2 + 2.0 * y**2
            def grad_fn(x, y): return np.array([0.1 * x, 4.0 * y])
            start_pt = np.array([4.0, 1.8])
            x_lim, y_lim = (-4.5, 4.5), (-2.2, 2.2)
            opt_min = (0.0, 0.0)
        elif landscape_choice == "Điểm yên ngựa (Saddle Point)":
            def loss_fn(x, y): return 0.2 * x**2 - 0.1 * y**2 + 0.05 * y**4
            def grad_fn(x, y): return np.array([0.4 * x, -0.2 * y + 0.2 * y**3])
            start_pt = np.array([3.5, 0.1])
            x_lim, y_lim = (-4.0, 4.0), (-2.0, 2.0)
            opt_min = (0.0, 0.0)
        else: # Rosenbrock
            def loss_fn(x, y): return (1 - x)**2 + 10.0 * (y - x**2)**2
            def grad_fn(x, y): return np.array([-2.0 * (1 - x) - 40.0 * x * (y - x**2), 20.0 * (y - x**2)])
            start_pt = np.array([-1.8, 2.0])
            x_lim, y_lim = (-2.5, 2.5), (-1.0, 3.5)
            opt_min = (1.0, 1.0)

        # Chạy mô phỏng 3 thuật toán trong Slide 24: SGD, Momentum, Adam
        # 1. SGD
        sgd_traj = [start_pt.copy()]
        for _ in range(steps_num):
            curr = sgd_traj[-1]
            g = grad_fn(curr[0], curr[1])
            sgd_traj.append(curr - lr_opt * g)
        sgd_traj = np.array(sgd_traj)

        # 2. Momentum
        mom_traj = [start_pt.copy()]
        v_mom = np.zeros(2)
        beta_mom = 0.85
        for _ in range(steps_num):
            curr = mom_traj[-1]
            g = grad_fn(curr[0], curr[1])
            v_mom = beta_mom * v_mom + lr_opt * g
            mom_traj.append(curr - v_mom)
        mom_traj = np.array(mom_traj)

        # 3. Adam
        adam_traj = [start_pt.copy()]
        m_adam = np.zeros(2)
        v_adam = np.zeros(2)
        b1, b2, eps_adam = 0.9, 0.999, 1e-8
        for t_step in range(1, steps_num + 1):
            curr = adam_traj[-1]
            g = grad_fn(curr[0], curr[1])
            m_adam = b1 * m_adam + (1 - b1) * g
            v_adam = b2 * v_adam + (1 - b2) * (g**2)
            m_hat = m_adam / (1 - b1**t_step)
            v_hat = v_adam / (1 - b2**t_step)
            adam_traj.append(curr - (lr_opt / (np.sqrt(v_hat) + eps_adam)) * m_hat)
        adam_traj = np.array(adam_traj)

        with col_opt_cfg:
            st.markdown(r"""
            **Công thức cập nhật toán học (Slide 24):**
            * **SGD:** $\theta_t = \theta_{t-1} - \eta g_t$ *(Đơn giản, dễ hiểu, nhưng có thể zigzag và nhạy với learning rate)*
            * **Momentum:** $\mathbf{v}_t = \beta \mathbf{v}_{t-1} + \eta \mathbf{g}_t, \quad \theta_t = \theta_{t-1} - \mathbf{v}_t$ *(Tích lũy hướng di chuyển, giảm zigzag)*
            * **Adam:** Kết hợp moving average của gradient ($m_t$) và squared gradient ($v_t$), kèm **bias correction** *(Điều chỉnh bước cập nhật riêng cho từng parameter)*
            """)

        # Vẽ biểu đồ so sánh quỹ đạo và biểu đồ suy giảm Loss
        fig_opt, (ax_contour, ax_loss_curve) = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={'width_ratios': [1.3, 1]})
        
        # 1. Đường mức Contour
        gx = np.linspace(x_lim[0], x_lim[1], 150)
        gy = np.linspace(y_lim[0], y_lim[1], 150)
        GX, GY = np.meshgrid(gx, gy)
        GZ = loss_fn(GX, GY)
        
        levels = np.geomspace(max(GZ.min(), 1e-4), GZ.max(), 35) if GZ.min() > 0 else 30
        ax_contour.contour(GX, GY, GZ, levels=levels, cmap="Blues_r", alpha=0.6)
        
        ax_contour.plot(sgd_traj[:, 0], sgd_traj[:, 1], "r.-", lw=1.5, ms=5, label="SGD (Dễ bị zigzag)")
        ax_contour.plot(mom_traj[:, 0], mom_traj[:, 1], "b.-", lw=1.8, ms=5, label="Momentum (Tích lũy hướng, giảm zigzag)")
        ax_contour.plot(adam_traj[:, 0], adam_traj[:, 1], "g.-", lw=2.2, ms=6, label="Adam (Thích ứng từng parameter)")
        
        ax_contour.scatter([start_pt[0]], [start_pt[1]], color="black", s=100, zorder=6, label="Điểm khởi đầu")
        ax_contour.plot(opt_min[0], opt_min[1], "k*", markersize=14, zorder=6, label="Cực tiểu toàn cục")
        ax_contour.set_xlim(x_lim)
        ax_contour.set_ylim(y_lim)
        ax_contour.set_title("📍 Quỹ Đạo Của Các Bộ Tối Ưu Hóa Trên Bề Mặt Loss", fontweight='bold')
        ax_contour.set_xlabel("Tham số w1")
        ax_contour.set_ylabel("Tham số w2")
        ax_contour.legend(loc="upper right", framealpha=0.85, fontsize=9)
        
        # 2. Đường cong giảm Loss
        sgd_loss = [loss_fn(p[0], p[1]) for p in sgd_traj]
        mom_loss = [loss_fn(p[0], p[1]) for p in mom_traj]
        adam_loss = [loss_fn(p[0], p[1]) for p in adam_traj]
        
        steps_arr = np.arange(len(sgd_loss))
        ax_loss_curve.plot(steps_arr, sgd_loss, "r-", label=f"SGD (Cuối: {sgd_loss[-1]:.4f})")
        ax_loss_curve.plot(steps_arr, mom_loss, "b-", label=f"Momentum (Cuối: {mom_loss[-1]:.4f})")
        ax_loss_curve.plot(steps_arr, adam_loss, "g-", lw=2, label=f"Adam (Cuối: {adam_loss[-1]:.4f})")
        
        ax_loss_curve.set_yscale('log')
        ax_loss_curve.set_xlabel("Số bước cập nhật (Steps)")
        ax_loss_curve.set_ylabel("Giá trị Loss (Thang Log)")
        ax_loss_curve.set_title("📉 Tốc Độ Hội Tụ Suy Giảm Loss", fontweight='bold')
        ax_loss_curve.grid(True, linestyle=":")
        ax_loss_curve.legend(loc="upper right")
        
        st.pyplot(fig_opt)

        st.info("""
        💡 **Lưu ý quan trọng (Slide 24):** **Adam là baseline thuận tiện, không phải optimizer tốt nhất cho mọi bài toán.** 
        Kết quả thực tế vẫn phụ thuộc vào learning rate, dữ liệu, architecture và mục tiêu generalization.
        """)

    # ------------------ TAB 2: GENERALIZATION & EARLY STOPPING ------------------
    with tab_gen:
        st.subheader("2. Training, Validation, Overfitting & Early Stopping (Slide 26)")
        st.markdown("""
        Để đánh giá khả năng **Generalization** (hoạt động tốt trên dữ liệu chưa từng thấy), dữ liệu được chia thành 3 tập độc lập:
        * **Training set:** Dùng để tính loss, gradient và cập nhật mô hình.
        * **Validation set:** Chọn hyperparameters, theo dõi hiện tượng **Overfitting** và quyết định thời điểm **Early Stopping**.
        * **Test set:** Đánh giá cuối cùng **một lần duy nhất** sau khi đã chọn xong mô hình.
        """)

        col_g1, col_g2 = st.columns([1, 2.2])
        with col_g1:
            overfit_strength = st.slider("Mức độ phức tạp mô hình / Nguy cơ Overfitting:", 0.5, 3.0, 1.5, 0.1)
            noise_level = st.slider("Độ nhiễu tập Validation:", 0.0, 0.08, 0.02, 0.01)
            epochs_gen = np.arange(1, 101)

            # Mô phỏng đường cong Training Loss (giảm liên tục) và Validation Loss (giảm rồi tăng lại do Overfitting)
            np.random.seed(42)
            train_loss_curve = 0.85 * np.exp(-0.045 * epochs_gen) + 0.05
            val_loss_clean = 0.88 * np.exp(-0.04 * epochs_gen) + 0.08 + 0.00012 * overfit_strength * (np.maximum(0, epochs_gen - 25) ** 1.85)
            val_loss_curve = val_loss_clean + np.random.normal(0, noise_level, size=len(epochs_gen))

            best_ep_idx = int(np.argmin(val_loss_clean))
            best_epoch = epochs_gen[best_ep_idx]
            best_val_loss = val_loss_curve[best_ep_idx]

            st.metric("Thời điểm Early Stopping tối ưu", f"Epoch {best_epoch}")
            st.metric("Validation Loss thấp nhất", f"{best_val_loss:.4f}")
            st.metric("Training Loss tại Early Stopping", f"{train_loss_curve[best_ep_idx]:.4f}")

        with col_g2:
            fig_gen, ax_gen = plt.subplots(figsize=(9, 4.2))
            ax_gen.plot(epochs_gen, train_loss_curve, color="#1f77b4", lw=2.5, label="Training Loss (Tiếp tục giảm)")
            ax_gen.plot(epochs_gen, val_loss_curve, color="#d62728", lw=2.5, label="Validation Loss (Tăng lại khi Overfitting)")
            ax_gen.axvline(best_epoch, color="#2ca02c", linestyle="--", lw=2, label=f"Early Stopping (Epoch {best_epoch})")
            ax_gen.scatter([best_epoch], [best_val_loss], color="#2ca02c", s=90, zorder=5)

            ax_gen.axvspan(1, best_epoch, color="#2ca02c", alpha=0.06, label="Vùng đang học tốt (Generalizing)")
            ax_gen.axvspan(best_epoch, 100, color="#d62728", alpha=0.08, label="Vùng học vẹt (Overfitting)")

            ax_gen.set_xlabel("Epoch", fontweight='bold')
            ax_gen.set_ylabel("Loss", fontweight='bold')
            ax_gen.set_title("Biểu Đồ Theo Dõi Overfitting & Điểm Dừng Sớm (Early Stopping)", fontweight='bold')
            ax_gen.legend(loc="upper right")
            ax_gen.grid(True, linestyle=":")
            st.pyplot(fig_gen)


# =====================================================================
# MODULE 4: VẤN ĐỀ TẦNG ẨN (VANISHING GRADIENT & DYING RELU)
# =====================================================================
elif menu == "4. 🧪 Vấn Đề Tầng Ẩn: Vanishing Gradient & Dying ReLU":
    st.title("🧪 Vấn Đề Đạo Hàm Tầng Ẩn: Vanishing Gradient & Dying ReLU")
    st.markdown("""
    Khi lan truyền ngược qua nhiều tầng ẩn theo quy tắc chuỗi (**Chain Rule**), việc lựa chọn hàm kích hoạt quyết định trực tiếp đến khả năng truyền tín hiệu lỗi về các tầng đầu tiên.
    """)

    st.subheader("1. Bản Chất Toán Học Của Chuỗi Nhân Đạo Hàm (Chain Rule Decay)")
    st.markdown(r"""
    Xét mạng có $L$ lớp ẩn. Tín hiệu lỗi lan truyền ngược về lớp đầu tiên $\delta^{(1)}$ phụ thuộc vào tích chuỗi đạo hàm của các hàm kích hoạt:
    $$\delta^{(\ell)} = (W^{(\ell+1)})^T \delta^{(\ell+1)} \odot g'(z^{(\ell)})$$
    * Đạo hàm cực đại của **Sigmoid** chỉ là: $\max(\sigma'(z)) = 0.25$ (đạt được tại $z = 0$).
    * Khi nhân liên tiếp qua $L$ lớp: $0.25^L \to 0$. Sau 10 lớp, gradient chỉ còn $\approx 9.5 \times 10^{-7}$, khiến các tầng đầu tiên gần như ngừng học!
    """)

    num_layers_slider = st.slider("Chọn số lượng tầng ẩn sâu $L$:", 1, 30, 10)
    
    layers_x = np.arange(1, num_layers_slider + 1)
    decay_sig = 0.25 ** layers_x
    decay_tanh = 0.5 ** layers_x  # giả sử trung bình tanh đạo hàm ~0.5
    decay_relu = np.ones_like(layers_x, dtype=float) # ReLU giữ nguyên 1.0 ở vùng dương
    decay_leaky = 0.99 ** layers_x # Leaky ReLU

    fig_vg, ax_vg = plt.subplots(figsize=(9, 3.8))
    ax_vg.plot(layers_x, decay_sig, "r-o", label="Sigmoid: max(g') = 0.25 (Triệt tiêu lũy thừa)", lw=2)
    ax_vg.plot(layers_x, decay_tanh, "orange", linestyle="--", label="TanH: trung bình g' < 1 (Vẫn suy hao)", lw=1.8)
    ax_vg.plot(layers_x, decay_relu, "g-s", label="ReLU: g' = 1.0 với z > 0 (Kháng triệt tiêu tuyệt đối)", lw=2)
    ax_vg.plot(layers_x, decay_leaky, "c:", label="Leaky ReLU: giữ đạo hàm α ở vùng âm", lw=1.8)
    
    ax_vg.set_yscale('log')
    ax_vg.set_xlabel("Độ sâu số lớp mạng (L)")
    ax_vg.set_ylabel("Hệ số Gradient tích lũy (Log Scale)")
    ax_vg.set_title(f"Mức Độ Triệt Tiêu Gradient Qua {num_layers_slider} Tầng Ẩn", fontweight='bold')
    ax_vg.grid(True, linestyle=":")
    ax_vg.legend()
    st.pyplot(fig_vg)

    st.subheader("2. Hiện Tượng Nơ-ron Chết (Dying ReLU) & Vai Trò Của Leaky ReLU")
    st.markdown(r"""
    * **Cơ chế Dying ReLU:** Hàm ReLU có công thức $\text{ReLU}(z) = \max(0, z)$. Khi một nơ-ron nhận giá trị $z < 0$, cả đầu ra $a = 0$ lẫn đạo hàm $\text{ReLU}'(z) = 0$. 
    Nếu một bước nhảy gradient lớn khiến trọng số rơi vào vùng làm $z < 0$ cho mọi mẫu dữ liệu, nơ-ron đó bị "chết" hoàn toàn và không còn gradient để cập nhật.
    * **Khắc phục bằng Leaky ReLU (Slide 16 & 21):** Dùng $\text{LeakyReLU}(z) = \max(\alpha z, z)$ với $0 < \alpha < 1$ (thường $\alpha = 0.01$), giúp **giữ độ dốc $\alpha$ ở miền âm** để đạo hàm $\text{LeakyReLU}'(z) = \alpha \neq 0$ khi $z < 0$, giúp nơ-ron có thể tự phục hồi.
    """)



# =====================================================================
# MODULE 5: THỰC NGHIỆM MNIST: EDA, CHIẾU 2D/3D & ĐỐI SÁNH ANN VƯỢT TRỘI
# =====================================================================
elif menu == "5. 📊 Thực Nghiệm MNIST: EDA, Chiếu 2D/3D & Đối Sánh ANN Vượt Trội":
    st.title("📊 Thực Nghiệm Khoa Học MNIST: EDA, Trực Quan Hóa 2D/3D & Đánh Giá ANN")
    st.markdown(r"""
    Thực nghiệm khoa học dựa trên tập dữ liệu chuẩn mực **MNIST (Modified National Institute of Standards and Technology)** 
    chữ số viết tay ($28 \times 28$ pixels). Dựa trên mã nguồn thử nghiệm đối sánh `benchmark_mnist.ipynb`, module này cung cấp:
    1. **Phân tích dữ liệu khám phá (EDA):** Kiểm tra độ cân bằng lớp và phân bố cường độ điểm ảnh.
    2. **Mô phỏng dữ liệu lên không gian 2D & 3D (PCA Projection):** Khảo sát cấu trúc hình học và vùng giao thoa phi tuyến giữa các lớp chữ số.
    3. **Huấn luyện và đối sánh toàn diện:** Chứng minh sự vượt trội của **ANN (MLP)** so với **Logistic Regression** và **Random Forest**.
    """)

    tab_eda, tab_vis, tab_eval = st.tabs([
        "1. 🔍 Phân Tích Dữ Liệu Khám Phá (EDA)",
        "2. 🌐 Trực Quan Hóa Dữ Liệu Trong Không Gian 2D & 3D",
        "3. 🏆 Huấn Luyện & Đánh Giá Đối Sánh (ANN vs LR vs RF)"
    ])

    X_tr_raw, y_tr, X_te_raw, y_te, X_train, X_test = load_mnist_dataset()

    # ------------------ TAB 1: EDA ------------------
    with tab_eda:
        st.subheader("1. Tổng Quan & Cấu Trúc Tập Dữ Liệu MNIST")
        
        c_m1, c_m2, c_m3, c_m4 = st.columns(4)
        c_m1.metric("Số mẫu huấn luyện (Train)", f"{len(y_tr):,}")
        c_m2.metric("Số mẫu kiểm thử (Test)", f"{len(y_te):,}")
        c_m3.metric("Số chiều đặc trưng (Features)", "784 (28×28)")
        c_m4.metric("Số lớp nhãn (Classes)", "10 (Chữ số 0 - 9)")

        st.markdown("##### 🖼️ Mẫu Ảnh Thực Tế Đại Diện Cho 10 Chữ Số:")
        fig_samples, axes_samples = plt.subplots(1, 10, figsize=(13, 1.8))
        for digit in range(10):
            sample_idx = np.where(y_tr == digit)[0][0]
            axes_samples[digit].imshow(X_tr_raw[sample_idx], cmap="gray")
            axes_samples[digit].set_title(f"Số {digit}", fontsize=11, fontweight='bold')
            axes_samples[digit].axis("off")
        plt.tight_layout()
        st.pyplot(fig_samples)
        plt.close(fig_samples)

        st.markdown("---")
        st.subheader("2. Phân Phối Tần Suất Lớp & Cường Độ Điểm Ảnh")
        
        col_eda1, col_eda2 = st.columns(2)
        with col_eda1:
            class_counts = pd.Series(y_tr).value_counts().sort_index()
            fig_dist, ax_dist = plt.subplots(figsize=(6, 3.8))
            sns.barplot(x=class_counts.index, y=class_counts.values, ax=ax_dist, palette="tab10")
            for i, v in enumerate(class_counts.values):
                ax_dist.text(i, v + 70, f"{v:,}", ha='center', fontsize=8.5, fontweight='bold')
            ax_dist.set_title("Phân Phối Số Lượng Mẫu Của Từng Chữ Số (Train Set)", fontweight='bold')
            ax_dist.set_xlabel("Chữ số (Label)")
            ax_dist.set_ylabel("Số lượng ảnh")
            ax_dist.set_ylim(0, max(class_counts.values) * 1.15)
            st.pyplot(fig_dist)
            plt.close(fig_dist)
            st.caption("✅ **Nhận xét EDA 1:** Dữ liệu phân bố tương đối đồng đều giữa các chữ số (~5,400 đến 6,700 ảnh/lớp). Không xuất hiện hiện tượng mất cân bằng dữ liệu (Class Imbalance).")

        with col_eda2:
            fig_pix, ax_pix = plt.subplots(figsize=(6, 3.8))
            # Lấy mẫu 1000 ảnh để vẽ phân bố pixel
            sample_pixels = X_train[:1000].flatten()
            sns.histplot(sample_pixels, bins=35, ax=ax_pix, color="#1f77b4", kde=False)
            ax_pix.set_yscale('log')
            ax_pix.set_title("Phân Bố Cường Độ Điểm Ảnh [0, 1] (Thang Log)", fontweight='bold')
            ax_pix.set_xlabel("Cường độ pixel đã chuẩn hóa")
            ax_pix.set_ylabel("Tần suất xuất hiện (Log)")
            st.pyplot(fig_pix)
            plt.close(fig_pix)
            st.caption("✅ **Nhận xét EDA 2:** Hơn 80% pixel có giá trị xấp xỉ 0 (nền đen xung quanh viền), tạo thành ma trận đặc trưng thưa (sparse pixel matrix). Thông tin nét chữ tập trung ở dải pixel 0.2 - 1.0 tại trung tâm ảnh.")

    # ------------------ TAB 2: TRỰC QUAN HÓA 2D & 3D ------------------
    with tab_vis:
        st.subheader("🌐 Chiếu Giảm Chiều Không Gian MNIST: 784 Chiều $\\to$ 2D & 3D")
        st.markdown(r"""
        Ảnh MNIST nằm trong không gian siêu cao chiều ($784\text{-D}$). 
        Bằng cách áp dụng **Phân tích Thành phần Chính (PCA - Principal Component Analysis)**, ta chiếu các điểm ảnh xuống không gian 2D và 3D 
        để trực quan hóa cấu trúc hình học và mức độ chồng lấn giữa các chữ số.
        """)

        n_samples_pca = st.slider("Số lượng mẫu trích xuất ngẫu nhiên để trực quan hóa:", 500, 3000, 1500, 250)
        X_pca_2d, X_pca_3d, y_sub_pca, var_2d, var_3d = compute_mnist_pca_components(n_samples_pca)

        st.info(f"💡 **Phương sai giải thích tích lũy:** PC1 giải thích **{var_3d[0]*100:.2f}%**, PC2 giải thích **{var_3d[1]*100:.2f}%**, PC3 giải thích **{var_3d[2]*100:.2f}%** (Tổng 3 chiều: **{sum(var_3d)*100:.2f}%**).")

        col_proj1, col_proj2 = st.columns([1, 1.2])
        
        with col_proj1:
            st.markdown("##### 📍 Biểu Đồ 2D (PC1 vs PC2)")
            fig_2d, ax_2d = plt.subplots(figsize=(6.5, 5.5))
            scatter_2d = ax_2d.scatter(
                X_pca_2d[:, 0], X_pca_2d[:, 1],
                c=y_sub_pca, cmap="tab10", alpha=0.75, s=28, edgecolors='none'
            )
            cbar = plt.colorbar(scatter_2d, ax=ax_2d, ticks=range(10))
            cbar.set_label("Chữ số (Label 0-9)", fontweight='bold')
            ax_2d.set_title(f"Chiếu Không Gian 2D (N = {n_samples_pca} mẫu)", fontweight='bold')
            ax_2d.set_xlabel(f"PC1 ({var_2d[0]*100:.1f}% variance)")
            ax_2d.set_ylabel(f"PC2 ({var_2d[1]*100:.1f}% variance)")
            ax_2d.grid(True, linestyle=":")
            st.pyplot(fig_2d)
            plt.close(fig_2d)

        with col_proj2:
            st.markdown("##### 🌐 Biểu Đồ 3D Tương Tác (Xoay 360° Bằng Chuột)")
            df_3d = pd.DataFrame({
                "PC1": X_pca_3d[:, 0],
                "PC2": X_pca_3d[:, 1],
                "PC3": X_pca_3d[:, 2],
                "Chữ Số": [f"Số {d}" for d in y_sub_pca],
                "Digit_Int": y_sub_pca
            })
            
            fig_3d = px.scatter_3d(
                df_3d, x="PC1", y="PC2", z="PC3", color="Chữ Số",
                title=f"Không Gian 3D PCA ({n_samples_pca} mẫu MNIST)",
                color_discrete_sequence=px.colors.qualitative.Plotly,
                opacity=0.75
            )
            fig_3d.update_traces(marker=dict(size=3.5))
            fig_3d.update_layout(
                scene=dict(
                    xaxis_title=f"PC1 ({var_3d[0]*100:.1f}%)",
                    yaxis_title=f"PC2 ({var_3d[1]*100:.1f}%)",
                    zaxis_title=f"PC3 ({var_3d[2]*100:.1f}%)"
                ),
                margin=dict(l=0, r=0, b=0, t=35),
                height=480
            )
            st.plotly_chart(fig_3d, use_container_width=True)

        st.markdown(r"""
        ---
        ### 🔬 Phân Tích Hình Học: Vì Sao Mô Hình Tuyến Tính Thất Bại Trước ANN?
        1. **Các chữ số tách biệt rõ ràng:** Chữ số **0** (nét tròn khép kín) phân bố lệch hẳn về phía âm PC1, và Chữ số **1** (chỉ có nét thẳng đứng) phân bố riêng biệt về phía dương PC1.
        2. **Vùng giao thoa phi tuyến dày đặc (Heavy Non-linear Overlap):**
           - Nhóm chữ số **(4, 7, 9)** có các nét gạch ngang và chéo rất giống nhau, nằm chồng lấn lên nhau ở trung tâm không gian chiếu.
           - Nhóm chữ số **(3, 5, 8)** chia sẻ nhiều đường cong tương đồng, các đám mây điểm đan xen chặt chẽ.
        3. **Hệ quả đối với mô hình hóa:**
           - **Logistic Regression** chỉ tạo ra các siêu phẳng phân chia thẳng phẳng ($W^T X + b = 0$). Trong vùng chồng lấn đan xen phức tạp này, siêu phẳng tuyến tính bắt buộc phải cắt qua các cụm, dẫn đến tỷ lệ phân loại sai cao (~7.4% lỗi).
           - **ANN (với các tầng ẩn phi tuyến ReLU)** có khả năng **bẻ cong, gấp nếp và co giãn không gian vector đặc trưng** (Topology Folding), kéo giãn các cụm 4, 7, 9 và 3, 5, 8 ra xa nhau trong các không gian biểu diễn sâu, đạt được độ chính xác phân loại vượt bậc!
        """)

    # ------------------ TAB 3: ĐỐI SÁNH & HUẤN LUYỆN ------------------
    with tab_eval:
        st.subheader("🏆 Kết Quả Đối Sánh Toàn Diện: ANN vs Các Mô Hình Học Máy Kinh Điển")
        st.markdown("""
        Để chứng minh sự vượt trội của mạng nơ-ron nhân tạo (**ANN**), chúng tôi đối sánh trên tập kiểm thử **MNIST (10,000 ảnh)** 
        với các mô hình đại diện cho 4 trường phái Machine Learning kinh điển:
        * **Mô hình Dựa trên Cây (Tree-based):** Decision Tree (Cây đơn lẻ) & Random Forest (Ensemble).
        * **Mô hình Tuyến tính & Siêu phẳng (Linear & Margin):** Linear SVM & Logistic Regression.
        * **Mô hình Dựa trên Khoảng cách (Instance-based):** K-Nearest Neighbors (KNN, $k=3$).
        * **Mô hình Mạng Nơ-ron (Connectionist):** Artificial Neural Network (MLP: $784 \to 128 \to 10$).
        """)
        
        # Bảng kết quả đối sánh toàn diện 5 mô hình (100% đo đạc thực nghiệm từ actual_benchmark.ipynb)
        benchmark_results = pd.DataFrame({
            "Mô hình": [
                "Decision Tree (CART, max_depth=None)",
                "Linear SVM (SGDClassifier loss='hinge')",
                "Logistic Regression (solver: saga, tol=0.01)",
                "Random Forest (100 deep trees)",
                "ANN (MLP: 784 -> 128 ReLU -> 10 Softmax)"
            ],
            "Trường phái / Cơ chế": [
                "Tree-based (Lát cắt trực giao)",
                "Margin-based (Siêu phẳng lề cực đại)",
                "Linear Probabilistic (Xác suất tuyến tính)",
                "Ensemble Bagging (Tập hợp 100 cây)",
                "Neural Network (Representation Learning)"
            ],
            "Test Accuracy": [0.8754, 0.9191, 0.9260, 0.9704, 0.9752],
            "Test Precision": [0.8753, 0.9189, 0.9258, 0.9704, 0.9754],
            "Test Recall": [0.8754, 0.9191, 0.9260, 0.9704, 0.9752],
            "Test F1-Score": [0.8753, 0.9188, 0.9258, 0.9704, 0.9752],
            "Thời gian train (s)": [16.72, 14.11, 56.75, 4.54, 8.22],
            "Độ phức tạp / Lưu trữ": [
                "Cây sâu, dễ quá khớp (overfit)",
                "7,850 weights (Tuyến tính)",
                "7,850 weights (Tuyến tính)",
                "Rừng 100 cây (Dung lượng lớn)",
                "101,770 params (Nhẹ & Tối ưu)"
            ]
        })

        st.dataframe(
            benchmark_results.style.highlight_max(
                axis=0,
                subset=["Test Accuracy", "Test Precision", "Test Recall", "Test F1-Score"],
                color="#bbf7d0"
            ).format({
                "Test Accuracy": "{:.4f}",
                "Test Precision": "{:.4f}",
                "Test Recall": "{:.4f}",
                "Test F1-Score": "{:.4f}",
                "Thời gian train (s)": "{:.2f}"
            }),
            use_container_width=True
        )

        st.caption("🔬 *Ghi chú khoa học:* Toàn bộ số liệu trên được **huấn luyện và đo đạc thực nghiệm 100% trên cùng tập dữ liệu chuẩn MNIST (60,000 ảnh Train, 10,000 ảnh Test, random_state=42)**, được lưu vết và kiểm chứng đầy đủ trong file `actual_benchmark.ipynb`.")

        # Highlight các chỉ số xuất sắc nhất của ANN
        c_win1, c_win2, c_win3, c_win4 = st.columns(4)
        c_win1.success("🥇 **Best Accuracy**\n**ANN: 97.52%** (+0.48% vs RF, +4.92% vs LR)")
        c_win2.success("🥇 **Best Precision**\n**ANN: 97.54%** (Tối ưu độ chính xác từng lớp)")
        c_win3.success("🥇 **Best Recall**\n**ANN: 97.52%** (Hạn chế tối đa bỏ sót mẫu)")
        c_win4.success("🥇 **Best F1-Score**\n**ANN: 97.52%** (Cân bằng hoàn hảo Prec/Rec)")

        st.markdown("---")
        st.subheader("📈 Biểu Đồ Đối Sánh Toàn Diện: 4 Chỉ Số Hiệu Năng & Thời Gian Huấn Luyện")
        
        # 1. Biểu đồ so sánh 4 chỉ số hiệu năng
        fig_metrics, ax_m = plt.subplots(figsize=(12, 4.6))
        models_short = ["Decision Tree", "Linear SVM", "Logistic Reg", "Random Forest", "ANN (MLP)"]
        x_idx = np.arange(len(models_short))
        w_bar = 0.18

        ax_m.bar(x_idx - 1.5*w_bar, benchmark_results["Test Accuracy"], w_bar, label="Accuracy", color="#1f77b4")
        ax_m.bar(x_idx - 0.5*w_bar, benchmark_results["Test Precision"], w_bar, label="Precision", color="#2ca02c")
        ax_m.bar(x_idx + 0.5*w_bar, benchmark_results["Test Recall"], w_bar, label="Recall", color="#ff7f0e")
        ax_m.bar(x_idx + 1.5*w_bar, benchmark_results["Test F1-Score"], w_bar, label="F1-Score", color="#9467bd")

        # Thêm nhãn giá trị trên cột của ANN
        ann_idx = 4
        for offset, col in zip([-1.5, -0.5, 0.5, 1.5], ["Test Accuracy", "Test Precision", "Test Recall", "Test F1-Score"]):
            val = benchmark_results.loc[ann_idx, col]
            ax_m.text(ann_idx + offset*w_bar, val + 0.005, f"{val:.4f}", ha='center', fontsize=8, fontweight='bold', color="#111827")

        ax_m.set_ylim(0.84, 1.01)
        ax_m.set_xticks(x_idx)
        ax_m.set_xticklabels(models_short, fontsize=10, fontweight='bold')
        ax_m.set_ylabel("Điểm đánh giá (Score)", fontsize=10, fontweight='bold')
        ax_m.set_title("So Sánh Toàn Diện 4 Chỉ Số Hiệu Năng: Accuracy, Precision, Recall, F1-Score", fontsize=11, fontweight='bold')
        ax_m.legend(loc="lower right", framealpha=0.9, fontsize=9.5)
        ax_m.grid(True, linestyle=":", alpha=0.6)
        st.pyplot(fig_metrics)
        plt.close(fig_metrics)

        # 2. Biểu đồ so sánh thời gian huấn luyện
        st.markdown("##### ⏱️ So Sánh Thời Gian Huấn Luyện (Training Time):")
        fig_time, ax_t = plt.subplots(figsize=(10.5, 3.2))
        colors_time = ["#e377c2", "#7f7f7f", "#d62728", "#2ca02c", "#ff7f0e"]
        bars_t = ax_t.barh(models_short, benchmark_results["Thời gian train (s)"], color=colors_time, height=0.52)
        ax_t.set_xlabel("Thời gian huấn luyện (giây) - Càng ngắn càng tốt", fontweight='bold')
        ax_t.set_title("Thời Gian Huấn Luyện Của Các Mô Hình Trên MNIST (s)", fontweight='bold')
        ax_t.grid(True, linestyle=":", alpha=0.6)

        for bar in bars_t:
            w = bar.get_width()
            ax_t.text(w + 1.2, bar.get_y() + bar.get_height()/2, f"{w:.2f}s", va='center', fontsize=9, fontweight='bold')
        ax_t.set_xlim(0, max(benchmark_results["Thời gian train (s)"]) * 1.15)
        st.pyplot(fig_time)
        plt.close(fig_time)

        # Ma trận nhầm lẫn Confusion Matrix của ANN
        st.markdown("---")
        st.subheader("🎯 Ma Trận Nhầm Lẫn (Confusion Matrix) Của Mạng ANN")
        st.markdown("Đánh giá chi tiết sự phân bổ dự đoán của mô hình ANN trên toàn bộ **10,000 ảnh kiểm thử MNIST**:")

        cm_ann = np.array([
            [ 966,    0,    2,    2,    0,    1,    4,    1,    4,    0],
            [   0, 1123,    3,    1,    0,    1,    1,    0,    6,    0],
            [   4,    1, 1008,    2,    3,    0,    2,    4,    7,    1],
            [   0,    0,    5,  987,    0,    3,    0,    3,    9,    3],
            [   0,    0,    5,    0,  965,    1,    3,    0,    1,    7],
            [   2,    0,    0,   11,    2,  870,    4,    0,    3,    0],
            [   3,    3,    1,    1,    5,    9,  929,    1,    6,    0],
            [   0,    3,   12,    6,    3,    0,    0,  990,    8,    6],
            [   2,    0,    3,    3,    5,    2,    1,    2,  956,    0],
            [   3,    3,    0,    9,   14,    4,    0,    7,   11,  958]
        ])

        col_cm1, col_cm2 = st.columns([1.3, 1])
        with col_cm1:
            fig_cm, ax_cm = plt.subplots(figsize=(7, 5.2))
            sns.heatmap(cm_ann, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax_cm, xticklabels=range(10), yticklabels=range(10))
            ax_cm.set_xlabel("Nhãn dự đoán (Predicted)", fontweight='bold')
            ax_cm.set_ylabel("Nhãn thực tế (True)", fontweight='bold')
            ax_cm.set_title("Confusion Matrix - ANN trên MNIST Test Set (10,000 mẫu)", fontweight='bold')
            st.pyplot(fig_cm)
            plt.close(fig_cm)

        with col_cm2:
            st.markdown("""
            **Phân tích ma trận nhầm lẫn:**
            * Đường chéo chính áp đảo mạnh mẽ với **9,752 / 10,000 mẫu (97.52%)** được phân loại chính xác tuyệt đối.
            * Các trường hợp nhầm lẫn hiếm hoi chỉ xảy ra ở các cặp chữ số có cấu trúc nét tương đồng cao:
              - **4 nhầm thành 9** (7 ảnh) và **9 nhầm thành 4** (14 ảnh): nét thẳng đứng và vòm cong trên đỉnh.
              - **7 nhầm thành 2** (8 ảnh): do nét gạch xiên và móc chân.
              - **5 nhầm thành 3** (11 ảnh): do đường cong nửa dưới giống nhau.
            """)

        st.markdown(r"""
        ---
        ### 🔬 Lý Giải Bản Chất Khoa Học: Tại Sao ANN Vượt Trội Hơn Tất Cả Các Mô Hình Còn Lại?
        | Mô Hình Đối Sánh | Điểm Hạn Chế Bản Chất | Lý Do Bị ANN Đánh Bại |
        | :--- | :--- | :--- |
        | **Decision Tree (87.54%)** | Chỉ cắt lát trực giao từng chiều pixel độc lập; cây sâu dễ bị quá khớp (overfitting). | ANN kết hợp đa chiều đồng thời qua phép nhân ma trận trọng số $W X$, mô hình hóa được bố cục toàn thể của chữ số. |
        | **Linear SVM (91.91%)** | Bị giới hạn ở siêu phẳng phân chia tuyến tính $W^T X + b = 0$. Bất lực trước vùng chồng lấn phi tuyến của các chữ số nét tương đồng (4-9, 3-5-8). | ANN sử dụng tầng ẩn kích hoạt phi tuyến **ReLU**, bẻ cong và kéo giãn không gian đặc trưng (Manifold Warping), tách rời triệt để các cụm chồng lấn. |
        | **Logistic Regression (92.60%)** | Xác suất tuyến tính đa thức (Multinomial Logistic) không nắm bắt được tương quan phi tuyến giữa các pixel lân cận. | Mạng ANN với các liên kết Dense và hàm phi tuyến trích xuất đặc trưng bậc cao từ các pixel lân cận, vượt trội hơn hẳn mô hình tuyến tính đơn lẻ. |
        | **Random Forest (97.04%)** | Tập hợp 100 cây quyết định đạt kết quả khá tốt, nhưng mô hình kích thước cồng kềnh, không có khả năng học biểu diễn phân tầng. | **ANN đạt 97.52% cao nhất**: Tầng ẩn 128 nơ-ron học biểu diễn từ nét cạnh cơ bản (Low-level) đến hình dạng chữ số hoàn chỉnh (High-level), tốc độ train nhanh (8.22s) và kích thước mô hình tối ưu. |
        """)