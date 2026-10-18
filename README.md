# CS313 / CS2207 — Khai Thác Dữ Liệu và Ứng Dụng (Data Mining and Application)

**Trường Đại Học Công Nghệ Thông Tin — ĐHQG-HCM (UIT - VNUHCM)**  
**Nhóm thực hiện (Nhóm 5):**
- Đặng Thị Thúy Hồng — `26848070`
- Đặng Anh Khoa — `26848099`
- Phạm Duy Khánh — `26848096`
- Võ Anh Kiệt — `26848109`
- Nguyễn Quan Liêm — `26848111`

---

## 📁 Cấu Trúc Thư Mục Dự Án

```text
CS313_CS2207_Data-Mining-And_Application/
├── midterm/                           # Báo cáo & Ứng dụng Demo Giữa Kỳ (Chuyên đề ANN)
│   ├── app.py                         # Ứng dụng tương tác Streamlit minh họa Toán học & Thực nghiệm ANN
│   ├── actual_benchmark.ipynb         # Notebook huấn luyện và đo đạc thực nghiệm trên tập dữ liệu MNIST
│   └── actual_benchmark_summary.csv   # Bảng tổng hợp kết quả thực nghiệm đối sánh các mô hình
├── final/                             # Đồ án Cuối Kỳ (Đang thực hiện - Xem Phần II bên dưới)
├── LICENSE
└── README.md                          # Tài liệu hướng dẫn và giải thích chi tiết dự án
```

---

# Phần I: Đồ Án Giữa Kỳ (`midterm/`)
## Chuyên Đề: Mạng Nơ-ron Nhân Tạo (ANN) — Từ Perceptron Đến Backpropagation

> **Câu hỏi xuyên suốt:** *"Làm thế nào những phép tính đơn giản có thể học được một ranh giới quyết định phức tạp?"*

Ứng dụng [`midterm/app.py`](midterm/app.py) được xây dựng bằng **Streamlit** nhằm trực quan hóa toàn bộ nền tảng toán học đằng sau mạng **Multilayer Perceptron (MLP)**: từ giới hạn của đường phân chia tuyến tính, cơ chế uốn cong không gian bằng hàm kích hoạt phi tuyến, sự kết hợp giữa hàm kích hoạt đầu ra với hàm mất mát (Loss Function), động lực học của các thuật toán tối ưu (Optimizer), khả năng tổng quát hóa (Generalization), cho đến thực nghiệm phân loại chữ số viết tay **MNIST**.

### 🚀 Hướng Dẫn Cài Đặt & Chạy Ứng Dụng Giữa Kỳ

1. **Cài đặt các thư viện phụ thuộc:**
   ```bash
   pip install streamlit numpy pandas matplotlib seaborn scikit-learn plotly tensorflow
   ```
2. **Khởi chạy ứng dụng Streamlit:**
   ```bash
   streamlit run midterm/app.py
   ```

---

### 🔬 Giải Thích Chi Tiết Kiến Trúc & 5 Module Trong `midterm/app.py`

#### 0. Lõi Toán Học Tự Hiện Thực (`InteractiveMLP` bằng NumPy thuần)

Thay vì dùng thư viện hộp đen (black-box) cho phần mô phỏng 2D, lớp `InteractiveMLP` trong [`midterm/app.py`](midterm/app.py) được cài đặt hoàn toàn bằng **NumPy** để có thể trích xuất và vẽ lại ranh giới quyết định (Decision Boundary) tại bất kỳ Epoch nào:

- **Khởi tạo trọng số (Weight Initialization):**
  - **Vì sao cần công thức khởi tạo riêng?** Nếu khởi tạo tất cả trọng số bằng $0$, mọi nơ-ron trong cùng tầng ẩn sẽ tính ra cùng một giá trị kích hoạt và nhận cùng một gradient (vấn đề *đối xứng trọng số - Symmetry Breaking*). Ngược lại, nếu khởi tạo ngẫu nhiên quá lớn hoặc quá nhỏ, tín hiệu sẽ phát nổ hoặc triệt tiêu ngay ở bước truyền thẳng.
  - **Công thức sử dụng:**
    - Với hàm kích hoạt `ReLU`: Dùng **khởi tạo He** vì ReLU triệt tiêu nửa miền âm nên cần nhân đôi phương sai để giữ nguyên năng lượng tín hiệu:

$$
\mathbf{W}^{(\ell)} \sim \mathcal{N}\left(0, \sqrt{\frac{2}{n_{\text{in}}}}\right)
$$

    - Với hàm kích hoạt `Tanh` / `Sigmoid`: Dùng **khởi tạo Xavier/Glorot**:

$$
\mathbf{W}^{(\ell)} \sim \mathcal{N}\left(0, \sqrt{\frac{1}{n_{\text{in}}}}\right)
$$

---

#### Module 1: 🎯 Live Demo — Nắn Đường Phân Loại Theo Epoch (Boundary Warping)

##### 1. Tại sao vấn đề xảy ra? (Giới hạn của mô hình tuyến tính)
- Một **Perceptron đơn lẻ (Single-layer Perceptron)** thực hiện phép tính:

$$
z = \mathbf{w}^T \mathbf{x} + b
$$

  Phương trình $\mathbf{w}^T \mathbf{x} + b = 0$ chỉ tạo ra **một siêu phẳng tuyến tính duy nhất** (một đường thẳng trong không gian 2D). Do đó, nó bất lực trước các bài toán **không khả phân tuyến tính (Non-linearly Separable)** như cổng logic **XOR**, dữ liệu hình hai vầng trăng khuyết (**Two Moons**) hoặc các đường tròn đồng tâm (**Circles**).

- Hơn nữa, nếu ghép nhiều tầng ẩn lại mà **không có hàm kích hoạt phi tuyến** (chỉ dùng phép biến đổi affine):

$$
\mathbf{h}_1 = \mathbf{W}_1 \mathbf{x} + \mathbf{b}_1, \quad \mathbf{h}_2 = \mathbf{W}_2 \mathbf{h}_1 + \mathbf{b}_2, \quad \mathbf{y} = \mathbf{W}_3 \mathbf{h}_2 + \mathbf{b}_3
$$

  Khi thế ngược vào nhau, toàn bộ mạng đa tầng rút gọn về $\mathbf{y} = (\mathbf{W}_3 \mathbf{W}_2 \mathbf{W}_1)\mathbf{x} + \mathbf{b}' = \mathbf{W}\mathbf{x} + \mathbf{b}'$, tức vẫn chỉ là một bộ phân loại tuyến tính!

##### 2. Sử dụng công thức nào & Tại sao giải quyết được?
- **Lan truyền thẳng (Forward Propagation) có phi tuyến:**

$$
\mathbf{z}^{(\ell)} = \mathbf{W}^{(\ell)} \mathbf{a}^{(\ell-1)} + \mathbf{b}^{(\ell)}, \quad \mathbf{a}^{(\ell)} = g^{(\ell)}\left(\mathbf{z}^{(\ell)}\right)
$$

  Với lớp đầu ra phân loại nhị phân:

$$
\hat{y} = \sigma\left(\mathbf{z}^{(L)}\right) = \frac{1}{1 + e^{-\mathbf{z}^{(L)}}}
$$

- **Bản chất hình học (Manifold Warping):**
  - Mỗi nơ-ron thứ $j$ ở lớp ẩn tạo ra một đường cắt tuyến tính $z_j = \mathbf{w}_j^T \mathbf{x} + b_j$.
  - Khi đi qua hàm phi tuyến $g(\cdot)$ như **ReLU** ($\max(0, z)$ — tạo nếp gấp tại $z=0$) hoặc **Tanh** ($\frac{e^z - e^{-z}}{e^z + e^{-z}}$ — uốn cong mềm hình chữ S), không gian đặc trưng ban đầu bị **bẻ gập và co giãn**.
  - Lớp đầu ra kết hợp tuyến tính các vùng không gian đã bị bẻ gập đó. Qua từng **Epoch** cập nhật trọng số bằng Backpropagation, đường đồng mức quyết định $\{\mathbf{x} \mid P(\hat{y}=1\mid\mathbf{x}) = 0.5\}$ được **"nắn" (warp)** từ một đường thẳng thô sơ thành đường cong uốn lượn ôm sát phân bố dữ liệu.

---

#### Module 2: 📐 Mô Phỏng Tương Tác — Hàm Loss & Xung Lực Gradient

##### 1. Tại sao vấn đề xảy ra? (Hiện tượng tê liệt Gradient khi dùng MSE cho phân loại)
- Giả sử ta dùng hàm mất mát bình phương trung bình **MSE (Mean Squared Error)** kết hợp với đầu ra **Sigmoid** $\hat{y} = \sigma(z)$ cho bài toán phân loại nhị phân:

$$
\mathcal{L}_{\text{MSE}} = \frac{1}{2}(y - \hat{y})^2
$$

- Áp dụng quy tắc chuỗi (**Chain Rule**), đạo hàm của hàm mất mát MSE theo đầu vào tuyến tính $z$ là:

$$
\frac{\partial \mathcal{L}_{\text{MSE}}}{\partial z} = \frac{\partial \mathcal{L}_{\text{MSE}}}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z} = (\hat{y} - y) \cdot \sigma'(z) = (\hat{y} - y)\hat{y}(1 - \hat{y})
$$

- **Nghịch lý toán học:** Khi mô hình **dự đoán sai nghiêm trọng nhất** (ví dụ nhãn thật $y = 1$, nhưng mô hình dự đoán $\hat{y} \approx 0.001$), lẽ ra mô hình phải nhận được một lực đẩy (gradient) cực lớn để sửa sai. Thế nhưng tại $\hat{y} \approx 0.001$, đạo hàm của Sigmoid rơi vào vùng bão hòa:

$$
\sigma'(z) = \hat{y}(1 - \hat{y}) \approx 0.001 \times 0.999 \approx 0.000999 \approx 0
$$

  Dẫn đến đạo hàm $\frac{\partial \mathcal{L}_{\text{MSE}}}{\partial z} \approx 0$. Mô hình bị **tê liệt hoàn toàn (Learning Slowdown)** dù đang đoán sai trầm trọng!

##### 2. Sử dụng công thức nào & Tại sao? (Nguyên lý phối hợp Output Activation + Loss)
- Để khắc phục, **Output Activation và Loss Function bắt buộc phải đi cặp với nhau**:

1. **Binary Classification (Sigmoid + Binary Cross-Entropy - BCE):**

$$
\mathcal{L}_{\text{BCE}} = -\left[y \ln(\hat{p}) + (1 - y)\ln(1 - \hat{p})\right]
$$

   Đạo hàm của hàm mất mát BCE theo xác suất dự đoán $\hat{p}$ là:

$$
\frac{\partial \mathcal{L}_{\text{BCE}}}{\partial \hat{p}} = -\frac{y}{\hat{p}} + \frac{1 - y}{1 - \hat{p}} = \frac{\hat{p} - y}{\hat{p}(1 - \hat{p})}
$$

   Khi nhân với đạo hàm của Sigmoid $\frac{\partial \hat{p}}{\partial z} = \hat{p}(1 - \hat{p})$ theo Chain Rule:

$$
\frac{\partial \mathcal{L}_{\text{BCE}}}{\partial z} = \frac{\hat{p} - y}{\hat{p}(1 - \hat{p})} \cdot \hat{p}(1 - \hat{p}) = \hat{p} - y
$$

   **Tại sao công thức này tối ưu?** Nhân tử gây bão hòa $\hat{p}(1-\hat{p})$ ở mẫu số và tử số **triệt tiêu hoàn toàn cho nhau**! Độ lớn xung lực gradient $|\hat{p} - y|$ tỷ lệ thuận trực tiếp với độ sai lệch: đoán sai càng xa ($|\hat{p} - y| \to 1$), lực đẩy sửa trọng số càng mạnh!

2. **Multiclass Classification (Softmax + Categorical Cross-Entropy - CCE):**

$$
\hat{p}_k = \frac{e^{z_k}}{\sum_{j=1}^K e^{z_j}}, \quad \mathcal{L}_{\text{CCE}} = -\sum_{k=1}^K y_k \ln(\hat{p}_k) \implies \frac{\partial \mathcal{L}_{\text{CCE}}}{\partial z_k} = \hat{p}_k - y_k
$$

---

#### Module 3: ⚡ Thuật Toán Tối Ưu (SGD, Momentum, Adam) & Generalization

Module này được chia làm 2 Tab tương ứng với bước **Cập nhật tham số (Optimization)** và **Đánh giá mô hình (Generalization)**:

##### Tab 1: Bộ Tối Ưu Hóa — SGD vs Momentum vs Adam
- **Tại sao vấn đề xảy ra với SGD tiêu chuẩn?**
  - Trên bề mặt hàm mất mát thực tế thường xuất hiện:
    1. **Hẻm núi dốc hẹp (Elongated Ravine):** $L(w_1, w_2) = 0.05 w_1^2 + 2.0 w_2^2$. Độ dốc theo hướng $w_2$ lớn gấp $40$ lần hướng $w_1$.
    2. **Điểm yên ngựa (Saddle Point):** $L(w_1, w_2) = 0.2 w_1^2 - 0.1 w_2^2 + 0.05 w_2^4$, nơi gradient tiệm cận $\mathbf{0}$ dù chưa phải cực tiểu.
    3. **Thung lũng cong Rosenbrock (Banana Valley):** $L(w_1, w_2) = (1 - w_1)^2 + 10(w_2 - w_1^2)^2$.
  - Thuật toán **SGD** cập nhật thuần túy theo gradient tức thời $\mathbf{g}_t = \nabla_\theta \mathcal{L}(\theta_{t-1})$:

$$
\theta_t = \theta_{t-1} - \eta \mathbf{g}_t
$$

  Tại hẻm núi hẹp, SGD bị lực đẩy lớn theo phương dốc đứng làm **dao động zigzag** văng qua lại hai vách núi, trong khi tiến rất chậm dọc theo đáy thung lũng về phía cực tiểu.

- **Sử dụng công thức cải tiến nào & Tại sao?**
  1. **Momentum (Quán tính):**

$$
\mathbf{v}_t = \beta \mathbf{v}_{t-1} + \eta \mathbf{g}_t, \quad \theta_t = \theta_{t-1} - \mathbf{v}_t
$$

     - *Lý giải:* Tích lũy trung bình động của các vector gradient quá khứ. Các thành phần gradient đổi dấu liên tục (phương ngang vách núi) sẽ tự triệt tiêu nhau, trong khi thành phần cùng chiều (dọc đáy thung lũng) được cộng hưởng giúp giảm hẳn zigzag và lướt qua vùng yên ngựa.

  2. **Adam (Adaptive Moment Estimation):**
     - Kết hợp **Moment bậc 1** (hướng di chuyển trung bình $\mathbf{m}_t$) và **Moment bậc 2** (trung bình bình phương gradient $\mathbf{v}_t$):

$$
\mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1 - \beta_1)\mathbf{g}_t, \quad \mathbf{v}_t = \beta_2 \mathbf{v}_{t-1} + (1 - \beta_2)\mathbf{g}_t^2
$$

     - Thực hiện **Hiệu chỉnh độ lệch (Bias Correction)** ở những bước đầu tiên và cập nhật tham số:

$$
\hat{\mathbf{m}}_t = \frac{\mathbf{m}_t}{1 - \beta_1^t}, \quad \hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1 - \beta_2^t}, \quad \theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon} \hat{\mathbf{m}}_t
$$

     - *Lý giải:* Việc chia cho $\sqrt{\hat{\mathbf{v}}_t}$ giúp Adam **tự động điều chỉnh bước nhảy riêng cho từng tham số**: chiều nào dốc đứng ($\hat{v}_t$ lớn) thì bước nhảy tự thu nhỏ lại để không văng bờ; chiều nào thoải ($\hat{v}_t$ nhỏ) thì bước nhảy tự mở rộng để hội tụ nhanh hơn. *(Lưu ý: Adam là một baseline thuận tiện, nhưng kết quả vẫn phụ thuộc vào learning rate, dữ liệu, kiến trúc và mục tiêu generalization).*

##### Tab 2: Khả Năng Generalization — Training, Validation & Early Stopping
- **Tại sao vấn đề xảy ra? (Hiện tượng Overfitting / Học vẹt)**
  - Khi mạng nơ-ron có nhiều tham số và được huấn luyện qua quá nhiều Epoch trên tập **Training**, mô hình không chỉ học quy luật tổng quát mà còn bắt đầu khớp vào cả nhiễu ngẫu nhiên (noise) của tập huấn luyện.
  - Hệ quả là `Training Loss` tiếp tục giảm sát về $0$, nhưng sai số trên dữ liệu mới (`Validation Loss`) lại bắt đầu **uốn cong đi lên**.
- **Giải pháp & Cơ chế trong App:**
  - Phân tách dữ liệu thành 3 tập: **Training set** (tính gradient cập nhật trọng số), **Validation set** (theo dõi độc lập sau mỗi epoch để chọn hyperparameter & dừng sớm), và **Test set** (chỉ đánh giá đúng 1 lần cuối cùng).
  - Áp dụng **Early Stopping**: Dừng quá trình huấn luyện ngay tại Epoch $t^*$ mà `Validation Loss` đạt cực tiểu:

$$
t^* = \arg\min_{t} \mathcal{L}_{\text{val}}(t)
$$

---

#### Module 4: 🧪 Vấn Đề Tầng Ẩn — Vanishing Gradient & Dying ReLU

##### 1. Tại sao xảy ra Vanishing Gradient (Triệt tiêu đạo hàm)?
- Trong giải thuật **Backpropagation**, tín hiệu lỗi $\delta^{(\ell)} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{(\ell)}}$ tại tầng ẩn thứ $\ell$ được truyền ngược từ tầng đầu ra $L$ về các tầng đầu tiên theo công thức đệ quy (**Chain Rule**):

$$
\delta^{(\ell)} = \left(\mathbf{W}^{(\ell+1)}\right)^T \delta^{(\ell+1)} \odot g'\left(\mathbf{z}^{(\ell)}\right)
$$

  Nếu khai triển cho tầng ẩn đầu tiên $\ell = 1$ trong mạng có $L$ tầng:

$$
\delta^{(1)} \propto \prod_{k=1}^{L-1} g'\left(\mathbf{z}^{(k)}\right)
$$

- **Nguyên nhân toán học:** Nếu dùng hàm kích hoạt **Sigmoid** ở các tầng ẩn, đạo hàm của nó là:

$$
\sigma'(z) = \sigma(z)(1 - \sigma(z)) \in (0, 0.25]
$$

  Giá trị lớn nhất của $\sigma'(z)$ chỉ bằng **$0.25$** (đạt tại $z = 0$). Khi nhân chuỗi liên tiếp qua $L$ tầng ẩn, hệ số suy giảm tỉ lệ với $0.25^L$. Ví dụ với $L = 10$ tầng ẩn:

$$
0.25^{10} \approx 9.53 \times 10^{-7} \approx 0
$$

  Gradient truyền về đến các tầng đầu tiên gần như bằng $0$, khiến các tầng trích xuất đặc trưng đầu tiên không thể cập nhật trọng số!

##### 2. Tại sao dùng ReLU lại sinh ra Dying ReLU, và khắc phục bằng công thức nào?
- **Giải pháp ReLU:** Hàm **ReLU** được định nghĩa là:

$$
\text{ReLU}(z) = \max(0, z), \quad \text{ReLU}'(z) = \begin{cases} 1, & z > 0 \\ 0, & z < 0 \end{cases}
$$

  Ở miền dương ($z > 0$), đạo hàm luôn bằng **$1$**, giúp nhân chuỗi qua nhiều tầng ($1^L = 1$) mà **không bị suy hao gradient**.
- **Vấn đề phát sinh (Dying ReLU):** Nếu một bước cập nhật lớn đẩy trọng số/bias của một nơ-ron sang vùng âm khiến $z < 0$ với mọi mẫu dữ liệu, nơ-ron đó sẽ xuất ra $a = 0$ và nhận đạo hàm $\text{ReLU}'(z) = 0$. Vì gradient bằng $0$, trọng số của nơ-ron đó không bao giờ được cập nhật nữa (nơ-ron bị "chết" vĩnh viễn).
- **Khắc phục bằng Leaky ReLU** (với $0 < \alpha < 1$, thường chọn $\alpha = 0.01$):

$$
\text{LeakyReLU}(z) = \max(\alpha z, z), \quad \text{LeakyReLU}'(z) = \begin{cases} 1, & z > 0 \\ \alpha, & z < 0 \end{cases}
$$

  Nhờ duy trì một độ dốc nhỏ $\alpha \neq 0$ ở miền âm, nơ-ron vẫn nhận được gradient khác $0$ để tự điều chỉnh và thoát khỏi vùng âm.

---

#### Module 5: 📊 Thực Nghiệm MNIST — EDA, Chiếu 2D/3D & Đối Sánh ANN Vượt Trội

Module này liên hệ lý thuyết ANN vào bài toán **Phân loại hình ảnh đa lớp (Multiclass Classification)** thực tế trong Khai thác Dữ liệu trên bộ dữ liệu chuẩn **MNIST** ($60,000$ ảnh huấn luyện và $10,000$ ảnh kiểm thử, kích thước $28 \times 28 = 784$ chiều, $K = 10$ lớp chữ số từ $0 \to 9$).

##### 1. Phân Tích Dữ Liệu Khám Phá (EDA) & Chiếu Không Gian PCA 2D/3D
- **Đặc trưng dữ liệu (EDA):** Phân phối số lượng mẫu giữa 10 lớp chữ số cân bằng (~$5,400$ đến $6,700$ mẫu/lớp), nhưng hơn $80\%$ điểm ảnh có cường độ bằng $0$ (nền đen), tạo ra không gian vector $784$ chiều rất thưa (sparse).
- **Phép chiếu giảm chiều PCA (Principal Component Analysis):**
  - Để quan sát cấu trúc hình học của dữ liệu $784$ chiều, ứng dụng tính ma trận hiệp phương sai và chiếu dữ liệu lên $k \in \{2, 3\}$ vector riêng (eigenvectors) ứng với các trị riêng lớn nhất:

$$
\mathbf{Z}_{\text{PCA}} = (\mathbf{X} - \boldsymbol{\mu})\mathbf{U}_k
$$

  - **Vì sao mô hình tuyến tính thất bại qua góc nhìn hình học 2D/3D?** Trên biểu đồ PCA 2D và 3D, chỉ có chữ số `0` và `1` tách rời tương đối rõ, trong khi nhóm chữ số có nét cong/chéo tương đồng như **`(4, 7, 9)`** và **`(3, 5, 8)`** đan xen, chồng lấn phi tuyến dày đặc ở vùng tâm. Mọi siêu phẳng tuyến tính ($\mathbf{W}\mathbf{x} + \mathbf{b} = \mathbf{0}$) bắt buộc phải cắt ngang qua các cụm này.

##### 2. Công Thức Kiến Trúc ANN & Các Độ Đo Đánh Giá Thực Nghiệm
- **Kiến trúc ANN (MLP) sử dụng:** `784 (Input) -> 128 (Hidden, ReLU) -> 10 (Output, Softmax)`, huấn luyện với hàm mất mát **Categorical Cross-Entropy (CCE)** và bộ tối ưu **Adam**.
- **Các chỉ số đánh giá (Evaluation Metrics):**

$$
\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}, \quad \text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}
$$

$$
\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}, \quad \text{F1-Score} = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
$$

- **Kết quả thực nghiệm đối sánh trên 10,000 ảnh MNIST Test Set** *(trích xuất từ [`midterm/actual_benchmark.ipynb`](midterm/actual_benchmark.ipynb))*:

| Mô Hình | Cơ Chế Học | Test Accuracy | Test Precision | Test Recall | Test F1-Score | Thời Gian Train (s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Decision Tree (CART)** | Chia cắt trực giao từng pixel độc lập | `87.54%` | `87.53%` | `87.54%` | `87.53%` | `16.72s` |
| **Linear SVM** | Siêu phẳng lề cực đại tuyến tính | `91.91%` | `91.89%` | `91.91%` | `91.88%` | `14.11s` |
| **Logistic Regression** | Phân loại xác suất tuyến tính | `92.60%` | `92.58%` | `92.60%` | `92.58%` | `56.75s` |
| **Random Forest (100 cây)** | Tập hợp Bagging 100 cây quyết định | `97.04%` | `97.04%` | `97.04%` | `97.04%` | `4.54s` |
| **ANN (MLP: 784 $\to$ 128 $\to$ 10)** | **Học biểu diễn phi tuyến (Representation Learning)** | **`97.52%`** | **`97.54%`** | **`97.52%`** | **`97.52%`** | `8.22s` |

- **Lý giải tại sao ANN đạt hiệu năng cao nhất:**
  - *So với Decision Tree (`87.54%`):* Cây quyết định rẽ nhánh theo từng ngưỡng pixel riêng lẻ ($x_i > \tau$), làm mất tính liên kết không gian 2 chiều của nét chữ và rất dễ bị Overfitting khi cây mọc sâu.
  - *So với Linear SVM (`91.91%`) & Logistic Regression (`92.60%`):* Cả hai đều bị giới hạn bởi biên quyết định tuyến tính phẳng, không thể tách được các chữ số chồng lấn phi tuyến như cặp `4 - 9` hoặc `3 - 5 - 8`.
  - *So với Random Forest (`97.04%`):* Nhờ tầng ẩn `128` nơ-ron kích hoạt phi tuyến `ReLU`, **ANN** tự động học được các đặc trưng tổ hợp bậc cao (từ các nét gạch, vòng cung cơ bản ghép thành hình dáng chữ số), giúp bẻ cong không gian biểu diễn để phân tách chính xác tới **9,752 / 10,000 mẫu (`97.52%`)** với kích thước tham số rất gọn nhẹ (`101,770` tham số).

---

# Phần II: Đồ Án Cuối Kỳ (`final/`)
> 📌 *Phần này được dành riêng để cập nhật nội dung, mã nguồn và báo cáo thực nghiệm cho Đồ Án Cuối Kỳ môn Khai Thác Dữ Liệu và Ứng Dụng.*

### 1. Giới Thiệu Bài Toán & Mục Tiêu (`TODO`)
- **Tên đề tài cuối kỳ:** *(Đang cập nhật...)*
- **Mô tả bài toán:** *(Đang cập nhật...)*
- **Mục tiêu khai thác dữ liệu:** *(Đang cập nhật...)*

### 2. Tập Dữ Liệu & Tiền Xử Lý (`TODO`)
- **Nguồn dữ liệu:** *(Đang cập nhật...)*
- **Quy trình làm sạch, trích xuất đặc trưng & EDA:** *(Đang cập nhật...)*

### 3. Phương Pháp Đề Xuất & Kiến Trúc Mô Hình (`TODO`)
- **Các thuật toán / Mô hình sử dụng:** *(Đang cập nhật...)*
- **Cơ sở toán học & Lý do lựa chọn:** *(Đang cập nhật...)*

### 4. Kết Quả Thực Nghiệm & Đánh Giá (`TODO`)
- **Kịch bản thực nghiệm & Độ đo đánh giá:** *(Đang cập nhật...)*
- **Bảng so sánh kết quả:** *(Đang cập nhật...)*

### 5. Hướng Dẫn Chạy Mã Nguồn Cuối Kỳ (`TODO`)
```bash
# Hướng dẫn chạy thực nghiệm và ứng dụng demo cuối kỳ tại thư mục final/
# (Sẽ cập nhật sau khi hoàn thành đồ án cuối kỳ)
```