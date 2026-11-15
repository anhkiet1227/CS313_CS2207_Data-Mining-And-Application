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
├── final/                             # Đồ án Cuối Kỳ (Chuyên đề DirectAU — KDD '22)
│   ├── app.py                         # Ứng dụng tương tác Streamlit gồm 6 module mô phỏng & gợi ý Top-K
│   ├── process_and_benmark.ipynb      # Notebook tải dữ liệu, tiền xử lý, huấn luyện & benchmark toàn diện
│   ├── dataset/                       # Dữ liệu gốc (.inter) từ nguồn thống nhất & các tập chia train/valid/test
│   ├── model/                         # Trọng số mô hình đã huấn luyện (.pt) cho Beauty, Gowalla, Yelp2018
│   └── benchmark_results/             # Bảng audit tiền xử lý, lịch sử huấn luyện, kết quả đánh giá & ablation
├── ref/                               # Bài báo gốc DirectAU.pdf và mã nguồn tham chiếu THUwangcy/DirectAU
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
## Chuyên Đề: DirectAU (KDD '22) — Towards Representation Alignment and Uniformity in Collaborative Filtering

> **Câu hỏi xuyên suốt:** *"Làm thế nào để tối ưu trực tiếp chất lượng hình học của không gian biểu diễn trên mặt cầu đơn vị trong Lọc cộng tác mà không cần lấy mẫu âm (Negative-Sample-Free)?"*

Dựa trên bài báo gốc *Towards Representation Alignment and Uniformity in Collaborative Filtering* (Chenyang Wang và cộng sự, KDD 2022 — [`ref/DirectAU.pdf`](ref/DirectAU.pdf)), phần đồ án cuối kỳ trong thư mục [`final/`](final/) hiện thực hóa hoàn chỉnh quy trình từ tải dữ liệu thô từ một nguồn thống nhất, tiền xử lý **Iterative 5-Core**, chia tập **80%/10%/10% theo từng User**, huấn luyện đối sánh **BPRMF** vs **DirectAU**, thực nghiệm triệt tiêu (**Ablation Study**), khảo sát độ nhạy siêu tham số $\gamma$, đến ứng dụng trực quan hóa & gợi ý tương tác **Streamlit** ([`final/app.py`](final/app.py)).

---

### 1. Giới Thiệu Bài Toán & Mục Tiêu
- **Tên đề tài cuối kỳ:** Hiện thực hóa, tái lập thực nghiệm và mô phỏng trực quan phương pháp **DirectAU** trong Hệ thống gợi ý Lọc cộng tác (Collaborative Filtering).
- **Vấn đề của phương pháp truyền thống (BPRMF):**
  - Hàm mục tiêu BPR phụ thuộc vào việc lấy mẫu âm ngẫu nhiên (negative sampling). Khi điểm dự đoán của cặp tương tác dương $(u, i^+)$ lớn hơn mẫu âm dễ $(u, i^-)$, đạo hàm $\sigma(s(u, i^-) - s(u, i^+)) \to 0$ làm quá trình kéo cặp dương lại gần bị chậm hoặc chững lại, đồng thời dễ gây suy giảm độ đồng đều trên mặt cầu biểu diễn.
- **Mục tiêu thực hiện trong [`final/process_and_benmark.ipynb`](final/process_and_benmark.ipynb) và [`final/app.py`](final/app.py):**
  1. Xây dựng pipeline tự động tải cả 3 bộ dữ liệu chuẩn (`Beauty`, `Gowalla`, `Yelp2018`) từ **1 nguồn thống nhất duy nhất**, kiểm định **Iterative 5-Core**, đối chiếu khớp **100%** với Bảng 1 của bài báo.
  2. Tách bạch rõ ràng dữ liệu (`final/dataset/`), trọng số mô hình (`final/model/`), và bảng thống kê/kết quả (`final/benchmark_results/`) với tiền tố đặt tên rõ ràng (`beauty_*`, `gowalla_*`, `yelp2018_*`).
  3. Huấn luyện & đánh giá **Full-Item Ranking** (`Recall@10, 20, 50` và `NDCG@10, 20, 50`) trên toàn bộ người dùng và toàn bộ catalog sản phẩm của cả 3 bộ dữ liệu.
  4. Kiểm chứng thực nghiệm **Ablation Study** (Section 5.3), **Độ nhạy tham số $\gamma$** (Section 5.6), hình học mặt cầu $\mathbb{S}^{63}$, và tích hợp vào giao diện tương tác 6 module trên Streamlit.

---

### 2. Tập Dữ Liệu & Tiền Xử Lý (`final/dataset/`)

Toàn bộ 3 bộ dữ liệu được tải bằng một hàm xử lý thống nhất `download_and_preprocess_dataset()` từ **cùng một nguồn chính thức** của nhóm tác giả DirectAU (`https://raw.githubusercontent.com/THUwangcy/DirectAU/main/dataset/`), kiểm tra schema, khử trùng lặp `(user_token, item_token)`, và lọc **Iterative 5-Core** (`filter_core(..., minimum=5)`):

| Bộ Dữ Liệu | Tệp Gốc Tải Về | Số Users | Số Items | Số Tương Tác (5-Core) | Min User Deg | Min Item Deg | Avg Inter / User | Mật Độ (Density) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Beauty** | `final/dataset/beauty.inter` | `22,363` | `12,101` | `198,502` | `5` | `5` | `8.88` | `0.0734%` |
| **Gowalla** | `final/dataset/gowalla.inter` | `29,858` | `40,981` | `1,027,370` | `9` | `10` | `34.41` | `0.0840%` |
| **Yelp2018** | `final/dataset/yelp2018.inter` | `31,668` | `38,048` | `1,561,406` | `18` | `10` | `49.31` | `0.1296%` |

- **Chia tập dữ liệu (`SPLIT_SEED = 2026`):** Áp dụng giao thức **80% Train / 10% Valid / 10% Test theo từng User** (`n_valid = max(1, int(0.1 * n))`, `n_test = max(1, int(0.1 * n))`), kiểm tra chặt chẽ `train ∩ valid = ∅`, `train ∩ test = ∅`, `valid ∩ test = ∅`:
  - **Beauty:** Train `148,766` | Valid `24,868` | Test `24,868`
  - **Gowalla:** Train `845,292` | Valid `91,039` | Test `91,039`
  - **Yelp2018:** Train `1,274,132` | Valid `143,637` | Test `143,637`
- **Tệp xuất ra tại `final/dataset/`:** Với mỗi bộ `<prefix>` $\in \{\text{beauty}, \text{gowalla}, \text{yelp2018}\}$, notebook tự động sinh đầy đủ: `<prefix>.inter`, `<prefix>_processed_interactions.csv`, `<prefix>_user_mapping.csv`, `<prefix>_item_mapping.csv`, `<prefix>_train.csv`, `<prefix>_valid.csv`, `<prefix>_test.csv`, `<prefix>_eval_users.npy`, và `dataset_source_info.json`.

---

### 3. Phương Pháp Đề Xuất & Kiến Trúc Mô Hình (`final/model/`)

1. **Bộ mã hóa (`Encoder`):**
   - Cài đặt **Matrix Factorization (`MF`)** với `EMBEDDING_DIM = 64`, khởi tạo `xavier_normal_`, cùng kiến trúc đồ thị **LightGCN (`LGCNEncoder`)** kiểm thử trong bộ unit test toán học `run_pipeline_checks()`.
2. **Hàm mục tiêu DirectAU (Eq. 6 & Eq. 7):**
   Chuẩn hóa $L_2$ các vector biểu diễn lên mặt cầu đơn vị $\mathbb{S}^{d-1}$ ($\tilde{f}(u) = \frac{f(u)}{\|f(u)\|_2}$, $\tilde{f}(i) = \frac{f(i)}{\|f(i)\|_2}$):
   $$l_{\text{align}} = \mathbb{E}_{(u, i) \sim p_{\text{pos}}} \|\tilde{f}(u) - \tilde{f}(i)\|_2^2$$
   $$l_{\text{uniform}} = \frac{1}{2} \log \mathbb{E}_{u, u' \sim p_{\text{user}}} e^{-2\|\tilde{f}(u) - \tilde{f}(u')\|_2^2} + \frac{1}{2} \log \mathbb{E}_{i, i' \sim p_{\text{item}}} e^{-2\|\tilde{f}(i) - \tilde{f}(i')\|_2^2}$$
   $$\mathcal{L}_{\text{DirectAU}} = l_{\text{align}} + \gamma \, l_{\text{uniform}}$$
   Trong đó, nhờ vector đã chuẩn hóa đơn vị ($\|\tilde{f}(x)\|_2 = 1$), biểu thức $-2\|\tilde{f}(x) - \tilde{f}(y)\|_2^2 = 4\langle \tilde{f}(x), \tilde{f}(y)\rangle - 4$ được tính trực tiếp qua tích ma trận đối xứng, đảm bảo tương đương toán học tuyệt đối trong cả bước lan truyền thẳng và đạo hàm ngược.
3. **Tối ưu hóa `ActiveRowAdam` & Lưu trữ Checkpoint:**
   - Sử dụng `ActiveRowAdam` chỉ cập nhật moment Adam cho các dòng Embedding xuất hiện trong mini-batch (tăng tốc đáng kể trên CPU và tránh trôi moment bậc 1/2 của các user/item không xuất hiện trong batch).
   - Trọng số tốt nhất (chọn theo `Validation NDCG@20`) được lưu riêng vào thư mục `final/model/`:
     - `beauty_BPR_best.pt`, `beauty_DirectAU_best.pt`
     - `gowalla_BPR_best.pt`, `gowalla_DirectAU_best.pt`
     - `yelp2018_BPR_best.pt`, `yelp2018_DirectAU_best.pt`

---

### 4. Kết Quả Thực Nghiệm & Đánh Giá (`final/benchmark_results/`)

#### A. Kết quả Full-Item Ranking trên toàn bộ Users & Catalog Items (`all_datasets_benchmark_results.csv`)
Cấu hình chạy trong [`final/process_and_benmark.ipynb`](final/process_and_benmark.ipynb): `EMBEDDING_DIM = 64`, `MAX_BATCHES_PER_EPOCH = 600`, `batch_size` (`256` cho Beauty; `512` cho Gowalla, Yelp2018), `12` epochs cho `BPR` (`lr=1e-3`) và `15` epochs cho `DirectAU` (`lr=3e-3`), đánh giá trên **100% users** và **toàn bộ catalog items** (mask lịch sử `train ∪ valid`):

| Bộ Dữ Liệu | Phương Pháp | $\gamma$ | Best Epoch | Recall@10 | Recall@20 | Recall@50 | NDCG@10 | NDCG@20 | NDCG@50 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Beauty** | `BPR` | — | `12/12` | `0.0467` | `0.0685` | `0.1030` | `0.0249` | `0.0306` | `0.0377` |
| **Beauty** | **`DirectAU`** | `0.5` | `14/15` | **`0.0834`** | **`0.1215`** | **`0.1833`** | **`0.0468`** | **`0.0567`** | **`0.0694`** |
| **Gowalla** | `BPR` | — | `12/12` | `0.0440` | `0.0648` | `0.1040` | `0.0319` | `0.0379` | `0.0476` |
| **Gowalla** | **`DirectAU`** | `1.0` | `12/15` | **`0.1131`** | **`0.1668`** | **`0.2675`** | **`0.0808`** | **`0.0962`** | **`0.1207`** |
| **Yelp2018** | `BPR` | — | `12/12` | `0.0244` | `0.0412` | `0.0771` | `0.0192` | `0.0248` | `0.0352` |
| **Yelp2018** | **`DirectAU`** | `2.0` | `15/15` | **`0.0484`** | **`0.0811`** | **`0.1497`** | **`0.0390`** | **`0.0500`** | **`0.0699`** |

#### B. Nghiên cứu Triệt tiêu (`beauty_ablation_history.csv`) & Độ nhạy $\gamma$ (`beauty_gamma_sensitivity.csv`)
- **Ablation Study trên Beauty (10 epochs):**
  - `align_only` ($\gamma = 0$): $l_{\text{align}}$ giảm từ `1.9952` xuống `0.2513`, nhưng $\exp(l_{\text{uniform}})$ tăng vọt gấp **13.4 lần** (từ `0.0238` lên `0.3180`), gây sụp đổ biểu diễn làm `Valid NDCG@20` đạt đỉnh `0.0446` ở Epoch 5 rồi tụt xuống `0.0267` ở Epoch 10.
  - `uniform_only`: Giữ $\exp(l_{\text{uniform}}) \approx 0.0209$ nhưng $l_{\text{align}}$ giữ nguyên quanh `2.0003`, khiến `Valid NDCG@20` chỉ đạt `0.0014` ở Epoch 10.
  - `directau` ($\gamma = 0.5$): Giữ độ đồng đều ổn định ($\exp(l_{\text{uniform}}) \approx 0.0236$) trong khi giảm $l_{\text{align}}$ xuống `0.7973`, giúp `Valid NDCG@20` tăng trưởng bền vững lên **`0.0517`** (Epoch 9) và `Valid Recall@20` đạt **`0.1141`** (Epoch 10).
- **Khảo sát độ nhạy $\gamma \in \{0.2, 0.5, 1.0, 2.0, 5.0\}$ trên Beauty (`eval_users`):**
  - Đỉnh hiệu năng đạt tại **$\gamma = 0.5$** (`Test Recall@20 = 0.1326`, `Test NDCG@20 = 0.0572`). Khi tăng $\gamma$ lên `5.0`, lực đẩy quá mạnh cản trở việc kéo các cặp tương tác dương lại gần ($l_{\text{align}}$ tăng lên `0.9854`), làm `Test Recall@20` giảm xuống `0.0854`.

#### C. Ứng dụng Tương tác Streamlit (`final/app.py`) gồm 6 Module
1. **Hình học Mặt cầu Đơn vị:** Mô phỏng tối ưu hóa trực tiếp trên $\mathbb{S}^1$ (2D) & $\mathbb{S}^2$ (3D xoay 360°) cho 4 chế độ và chứng minh Định lý 1.
2. **Tiền xử lý Dữ liệu:** Nhật ký tải từ nguồn thống nhất, đối chiếu Table 1, chia tập 80/10/10 và phân phối bậc Long-Tail trên cả 3 tập.
3. **Động lực Học tập:** Đường cong hội tụ `NDCG@20`, `Recall@20`, quỹ đạo $(l_{\text{uniform}}, l_{\text{align}})$ thực nghiệm và bản đồ 8 phương pháp CF (Figure 4).
4. **Ablation Study & Độ nhạy $\gamma$:** Biểu đồ minh họa hiện tượng Dimensional Collapse và quy luật phụ thuộc mật độ dữ liệu của $\gamma$.
5. **Đối sánh Toàn diện:** Kết quả thực nghiệm tự chạy trên 3 bộ dữ liệu, phân phối góc Cosine trên $\mathbb{S}^{63}$, và bảng đối chiếu Tables 2, 3, 4 của bài báo.
6. **Hệ thống Gợi ý Top-K Thực tế:** Nạp checkpoint trực tiếp từ `final/model/` để gợi ý và kiểm tra Hit/Recall/NDCG theo thời gian thực cho cả **Beauty**, **Gowalla**, và **Yelp2018**.

---

### 5. Hướng Dẫn Chạy Mã Nguồn Cuối Kỳ (`final/`)

1. **Cài đặt các thư viện cần thiết:**
   ```bash
   pip install torch numpy pandas scipy matplotlib plotly streamlit
   ```
2. **Chạy toàn bộ Notebook tải dữ liệu, tiền xử lý, huấn luyện & xuất kết quả:**
   Mở file [`final/process_and_benmark.ipynb`](final/process_and_benmark.ipynb) và bấm **Run All**. Notebook sẽ tự động tạo toàn bộ tệp dữ liệu trong `final/dataset/`, trọng số mô hình trong `final/model/`, và bảng kết quả trong `final/benchmark_results/`.
3. **Khởi chạy ứng dụng Streamlit cuối kỳ:**
   ```bash
   streamlit run final/app.py
   ```