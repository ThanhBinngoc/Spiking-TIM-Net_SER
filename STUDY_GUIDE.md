# BÁO CÁO TOÀN DIỆN VỀ KIẾN TRÚC VÀ CƠ CHẾ HOẠT ĐỘNG CỦA MÔ HÌNH
# SPIKING-TIM-NET (SPEECH EMOTION RECOGNITION)

> **Tài liệu học tập & Nghiên cứu chuyên sâu**  
> **Đề tài:** Nhận dạng Cảm xúc Tiếng nói sử dụng mạng Mẫu hình Thời gian Hai chiều Đa thang đo kết hợp Cơ chế Nơ-ron Xung Sinh học (Spiking-TIM-Net).  
> **Tác giả:** ThanhBinngoc  

---

## MỤC LỤC CHI TIẾT
1. [Bức tranh Tổng thể & Luồng Dữ liệu (End-to-End Data Flow)](#1-bức-tranh-tổng-thể--luồng-dữ-liệu-end-to-end-data-flow)
2. [Giai đoạn 1: Input Pipeline & Trích xuất Đặc trưng MFCC](#2-giai-đoạn-1-input-pipeline--trích-xuất-đặc-trưng-mfcc)
3. [Giai đoạn 2: Tách Hai Luồng Thời Gian Xuôi - Ngược (Bi-directional Stream)](#3-giai-đoạn-2-tách-hai-luồng-thời-gian-xuôi---ngược-bi-directional-stream)
4. [Giai đoạn 3: Khối Cốt Lõi - Temporal-Aware Block (TAB)](#4-giai-đoạn-3-khối-cốt-lõi---temporal-aware-block-tab)
5. [Giai đoạn 4: Điểm mới - Cơ chế Nơ-ron Xung LIF (Spiking Attention at F)](#5-giai-đoạn-4-điểm-mới---cơ-chế-nơ-ron-xung-lif-spiking-attention-at-f)
6. [Giai đoạn 5: Hợp nhất Đa Thang Đo Thời Gian (Dynamic Multi-Scale Fusion)](#6-giai-đoạn-5-hợp-nhất-đa-thang-đo-thời-gian-dynamic-multi-scale-fusion)
7. [Giai đoạn 6: Phân loại Cảm xúc Đầu ra (Classifier)](#7-giai-đoạn-6-phân-loại-cảm-xúc-đầu-ra-classifier)
8. [Chiến lược Huấn luyện & Đánh giá (Training Strategy)](#8-chiến-lược-huấn-luyện--đánh-giá-training-strategy)
9. [Bảng tóm tắt Kích thước Tensor (Shape Tracking) qua từng tầng](#9-bảng-tóm-tắt-kích-thước-tensor-shape-tracking-qua-từng-tầng)

---

## 1. BỨC TRANH TỔNG THỂ & LUỒNG DỮ LIỆU (END-TO-END DATA FLOW)

Mô hình hoạt động theo nguyên lý biến đổi một đoạn tín hiệu âm thanh liên tục (Continuous Audio Waveform) thành phân phối xác suất của 8 cảm xúc rời rạc thông qua sơ đồ khối dưới đây:

```
[Audio Waveform (.wav)] 
       │ (Tần số lấy mẫu 22.050 Hz)
       ▼
┌────────────────────────────────────────────────────────┐
│ GIAI ĐOẠN 1: INPUT PIPELINE                            │
│ - Chuẩn hóa độ dài: 110.000 mẫu (~5s cho RAVDESS)      │
│ - Trích xuất 39 đặc trưng tĩnh MFCC                    │
│ ➔ Ma trận đặc trưng: Shape (Batch, 215, 39)           │
└────────────────────────────────────────────────────────┘
       │
       ├────────────────────────────────────────┐
       ▼ (Luồng Xuôi)                           ▼ (Luồng Ngược)
┌──────────────────────────────┐         ┌──────────────────────────────┐
│ GIAI ĐOẠN 2A: FORWARD STREAM │         │ GIAI ĐOẠN 2B: REVERSE STREAM │
│ - 1x1 Conv chiếu kênh        │         │ - Đảo ngược thời gian        │
│ - 8 khối TAB (Dilation: 1→128)         │ - 1x1 Conv chiếu kênh        │
│ ➔ Đặc trưng xuôi: F_fwd_i    │         │ - 8 khối TAB (Dilation: 1→128)
└──────────────┬───────────────┘         │ ➔ Đặc trưng ngược: F_bwd_i   │
               │                         └──────────────┬───────────────┘
               └───────────────────┬────────────────────┘
                                   │ (Cộng hai chiều: F_fwd_i ⊕ F_bwd_i)
                                   ▼
┌────────────────────────────────────────────────────────┐
│ GIAI ĐOẠN 3: TỔNG HỢP ĐA THANG ĐO (DYNAMIC FUSION)      │
│ - Global Average Pooling tạo 8 vector đại diện {g_i}   │
│ - WeightLayer: Tự học trọng số kết hợp w_i             │
│ ➔ Vector ngữ cảnh toàn cục: Shape (Batch, 39)          │
└────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────┐
│ GIAI ĐOẠN 4: CLASSIFIER (PHÂN LOẠI)                    │
│ - Fully Connected (Dense 39 ➔ 8)                       │
│ - Softmax: Xuất xác suất của 8 cảm xúc                 │
└────────────────────────────────────────────────────────┘
```

---

## 2. GIAI ĐOẠN 1: INPUT PIPELINE & TRÍCH XUẤT ĐẶC TRƯNG MFCC

### 2.1. Bản chất vật lý của tín hiệu âm thanh
* File âm thanh `.wav` lưu trữ biên độ dao động sóng âm theo thời gian với tần số lấy mẫu $fs = 22.050\text{ Hz}$ (tức mỗi giây ghi lại 22.050 con số thực).
* **Vấn đề:** Dữ liệu sóng âm thô (Waveform) có kích thước quá lớn, chứa nhiều thông tin pha và dao động vi mô không phản ánh trực tiếp cảm xúc của con người.

### 2.2. Đồng bộ hóa độ dài tín hiệu (Length Normalization)
Mỗi câu nói có độ dài khác nhau. Mạng nơ-ron bắt buộc phải có kích thước đầu vào đồng nhất:
* Trong code ([`extract_feature.py`](file:///home/thanhbinh/Downloads/TIM-Net_SER-main/Code/extract_feature.py)), tác giả quy định `mean_signal_length = 110.000` mẫu đối với tập **RAVDESS** (tương đương khoảng $5.0$ giây).
* **Nếu câu ngắn hơn 110.000 mẫu:** Đệm thêm các số $0$ cân đối vào hai đầu (`Zero-padding`).
* **Nếu câu dài hơn 110.000 mẫu:** Cắt bớt phần thừa ở hai đầu, giữ lại đoạn âm thanh ở giữa.

### 2.3. Quy trình toán học trích xuất MFCC 39 chiều
1. **Phân khung (Framing):** Chia tín hiệu thành các khung nhỏ có độ dài $50\text{ ms}$ (tương ứng 1102 mẫu) với bước trượt $12.5\text{ ms}$ (275 mẫu).
   * *Ý nghĩa vật lý:* Tín hiệu tiếng nói biến thiên liên tục (non-stationary), nhưng trong khoảng cực ngắn $50\text{ ms}$, cấu trúc thanh quản con người được coi là ổn định cục bộ (quasi-stationary).
2. **Cửa sổ hóa (Windowing):** Nhân từng khung với hàm cửa sổ Hamming $w(n) = 0.54 - 0.46 \cos\left(\frac{2\pi n}{N-1}\right)$ để làm mượt biên hai đầu, tránh hiện tượng rò rỉ phổ tần số (spectral leakage).
3. **Biến đổi Fourier nhanh (STFT/FFT):** Chuyển từng khung từ miền thời gian sang miền tần số, tính toán phổ công suất (Power Spectrum).
4. **Dải lọc thang đo Mel (Mel Filterbank):** Tai người nghe âm thanh phi tuyến tính (nhạy với âm trầm hơn âm bổng). Ta chiếu năng lượng qua dải lọc Mel:
   $$m = 2595 \log_{10}\left(1 + \frac{f}{700}\right)$$
5. **Log Năng lượng & Biến đổi Cosin rời rạc (DCT):** Nén năng lượng phổ và giữ lại đúng **39 hệ số đầu tiên** ($n\_mfcc = 39$).
6. **Chuyển vị ma trận (Transpose):** Kết quả ban đầu từ thư viện librosa có dạng $(39, 215)$. Sau khi chuyển vị `np.transpose`, ma trận có dạng chuẩn:
   $$\mathbf{X} \in \mathbb{R}^{215 \times 39}$$
   *(215 bước thời gian, 39 đặc trưng tần số).*

### 2.4. Ý nghĩa biểu đồ t-SNE `Class-Confusion` (Góc dưới bên trái)
Khi chiếu các vector MFCC ban đầu xuống không gian 2 chiều bằng thuật toán t-SNE:
* Tất cả 8 màu đại diện cho 8 cảm xúc bị **trộn lẫn hoàn toàn (Entangled)**.
* **Kết luận khoa học:** Bản thân đặc trưng MFCC tĩnh hoàn toàn không thể phân loại được cảm xúc nếu không có kiến trúc bóc tách ngữ cảnh thời gian ở phía sau.

---

## 3. GIAI ĐOẠN 2: TÁCH HAI LUỒNG THỜI GIAN XUÔI - NGƯỢC (BI-DIRECTIONAL STREAM)

### 3.1. Tại sao phải xử lý hai chiều thời gian?
Trong giao tiếp, cảm xúc của câu nói không chỉ phụ thuộc vào những từ đã nói trước đó, mà ngữ điệu ở **cuối câu** (ví dụ: lên giọng bất ngờ, ngắt giọng đột ngột) thường quyết định tính chất cảm xúc của toàn bộ câu. Xử lý cả hai chiều giúp mạng nắm bắt trọn vẹn cả **ngữ cảnh quá khứ lẫn tương lai**.

### 3.2. Triển khai trong mã nguồn ([`Code/TIMNET.py`](file:///home/thanhbinh/Downloads/TIM-Net_SER-main/Code/TIMNET.py))
* **Luồng xuôi (Forward):** Nhận trực tiếp đầu vào $X$ có shape $(B, 215, 39)$.
* **Luồng ngược (Backward):** Đảo ngược theo trục thời gian bằng tầng Lambda:
  ```python
  backward = Lambda(lambda x: K.reverse(x, axes=1))(inputs)
  ```
* **Tầng chiếu khởi tạo $1\times 1$ Conv:** Cả hai luồng đi qua một tầng tích chập $1\times 1$ để chuẩn hóa số kênh về `nb_filters = 39`:
  ```python
  forward_convd = Conv1D(filters=39, kernel_size=1, padding='causal')(forward)
  backward_convd = Conv1D(filters=39, kernel_size=1, padding='causal')(backward)
  ```

---

## 4. GIAI ĐOẠN 3: KHỐI CỐT LÕI - TEMPORAL-AWARE BLOCK (TAB)

Khối TAB là trái tim của kiến trúc TIM-Net, được lặp lại 8 lần tương ứng với 8 mức độ giãn nở thời gian:

$$\vec{F}_{n-1} \longrightarrow \underbrace{[\text{DC Conv} \to \text{BN} \to \text{ReLU} \to \text{SD}] \times 2}_{\text{Trích xuất chuỗi thời gian}} \longrightarrow \sigma \longrightarrow \odot \vec{F}_{n-1} \longrightarrow \vec{F}_n$$

### 4.1. Tích chập Giãn nở Nhân quả (Dilated Causal Convolution - DC Conv)
* **Luật nhân quả (Causal):** Đầu ra tại thời điểm $t$ chỉ được tính toán dựa trên các thời điểm hiện tại và quá khứ ($\le t$), hoàn toàn không bị rò rỉ thông tin tương lai.
* **Hệ số giãn nở (Dilation Rate $d$):** Ở khối thứ $i$ ($i = 1, \dots, 8$), hệ số giãn nở tăng theo cấp số nhân:
  $$d_i = 2^{i-1} \in \{1, 2, 4, 8, 16, 32, 64, 128\}$$
* **Cơ chế hoạt động:** Thay vì trượt kernel trên các khung liền kề, tích chập giãn nở bỏ cách $d-1$ điểm dữ liệu.
* **Vùng tiếp nhận (Receptive Field - RF):**
  $$RF_i = 1 + \sum_{k=1}^i (K - 1) \cdot d_k$$
  *(Với kích thước kernel $K = 2$).*
  * Ở khối 1 ($d=1$): Mạng chỉ nhìn thấy ngữ cảnh cực ngắn (vài phần mười mili-giây, cấp độ âm vị).
  * Ở khối 8 ($d=128$): Vùng tiếp nhận mở rộng bao phủ toàn bộ 215 khung thời gian ($5$ giây câu nói).
  * **Ưu điểm vượt trội:** Mở rộng vùng quan sát cực đại mà số lượng tham số mô hình **không tăng thêm một tham số nào**!

### 4.2. Chuẩn hóa Lô (Batch Normalization - BN)
Chuẩn hóa giá trị kích hoạt qua từng batch, giúp giảm thiểu hiện tượng dịch chuyển hiệp biến nội tại (Internal Covariate Shift), cho phép mạng hội tụ nhanh và ổn định.

### 4.3. Kích hoạt Phi tuyến ReLU
Giữ nguyên giá trị dương và triệt tiêu giá trị âm, giúp mạng học được các hàm quan hệ phi tuyến tính phức tạp trong đặc trưng âm thanh.

### 4.4. Spatial Dropout 1D (SD)
* Khác với Dropout thông thường (chỉ tắt ngẫu nhiên từng nơ-ron đơn lẻ), **Spatial Dropout 1D tắt ngẫu nhiên nguyên cả một kênh đặc trưng trên toàn bộ 215 bước thời gian** (tỷ lệ tắt $10\%$).
* **Lý do sử dụng:** Trong âm thanh, các khung thời gian sát nhau có tương quan rất cao. Nếu chỉ tắt 1 điểm nơ-ron, mô hình vẫn dễ dàng nhìn lén thông tin từ điểm bên cạnh. Tắt toàn bộ kênh ép mô hình phải học phân tán đều, chống học vẹt (overfitting) cực kỳ hiệu quả.

---

## 5. GIAI ĐOẠN 4: ĐIỂM MỚI - CƠ CHẾ NƠ-RON XUNG LIF (SPIKING ATTENTION AT F)

Đây là **đóng góp khoa học trọng tâm** của đề tài cải tiến.

### 5.1. So sánh giữa Bản gốc và Bản cải tiến Spiking

* **Bản gốc (Sigmoid Gating mềm):**
  $$A = \sigma(\text{Conv}) \in [0.0, 1.0]$$
  $$F = \vec{F}_{n-1} \odot A$$
  *Điểm yếu:* Giá trị $A$ luôn là số thực liên tục ($0.1, 0.2, 0.3,...$). Do đó, các đoạn khoảng lặng, tiếng thở, tạp âm nền vẫn giữ lại một phần năng lượng đi vào các tầng sau, làm loãng đặc trưng cảm xúc.

* **Bản cải tiến (Spike-driven Gating nhị phân):**
  $$S = \text{LIF\_Spike}(A) \in \{0.0, 1.0\}$$
  $$F = \vec{F}_{n-1} \odot S$$
  *Ưu thế vượt trội:* Cổng trở thành một chiếc công tắc nhị phân. Các đoạn không chứa cảm xúc bị **dập tắt tuyệt đối về 0**, chỉ những khoảnh khắc năng lượng cảm xúc bộc lộ mạnh mẽ mới kích hoạt **xung 1** cho tín hiệu đi qua.

### 5.2. Phương trình toán học Nơ-ron LIF (Leaky Integrate-and-Fire)
Mô phỏng chính xác màng tế bào thần kinh sinh học:
1. **Tích lũy & Rò rỉ điện thế (Leaky Integration):**
   $$V(t) = \tau \cdot V(t-1) \cdot (1 - S(t-1)) + I(t)$$
   * $I(t) = \sigma(\text{Conv}_t)$: Dòng kích thích đầu vào từ tầng tích chập.
   * $\tau = 0.85$: Hệ số rò rỉ điện thế màng theo thời gian.
   * $(1 - S(t-1))$: Cơ chế **Hard Reset** — nếu bước trước nơ-ron vừa bắn xung ($S=1$), điện thế lập tức xả về 0.
2. **Kích hoạt phát xung (Firing Mechanism):**
   $$S(t) = \text{Heaviside}(V(t) - V_{th}) = \begin{cases} 1.0 & \text{khi } V(t) \ge V_{th} \quad (V_{th} = 0.5) \\ 0.0 & \text{khi } V(t) < V_{th} \end{cases}$$

### 5.3. Kỹ thuật Đạo hàm Đại diện (Surrogate Gradient)
Hàm bước nhảy Heaviside có đạo hàm bằng 0 ở khắp mọi nơi (không khả vi), khiến thuật toán lan truyền ngược (Backpropagation) không thể cập nhật trọng số. Ta giải quyết bằng hàm **Fast Sigmoid Surrogate Gradient**:
* **Khi truyền xuôi (Forward):** Vẫn bắn xung nhị phân $0$ hoặc $1$ chuẩn xác.
* **Khi truyền ngược (Backward):** Đạo hàm được tính xấp xỉ bằng:
  $$\frac{\partial S}{\partial V} \approx \frac{1}{(1 + \alpha |V - V_{th}|)^2} \quad (\alpha = 10.0)$$
  Giúp dòng gradient chảy mượt mà qua nơ-ron xung về các tầng trước để cập nhật trọng số.

---

## 6. GIAI ĐOẠN 5: HỢP NHẤT ĐA THANG ĐO THỜI GIAN (DYNAMIC MULTI-SCALE FUSION)

Sau khi đi qua 8 khối TAB ở 2 luồng:
1. Tại mỗi thang đo $i \in \{1, \dots, 8\}$, đặc trưng xuôi và ngược được cộng gộp:
   $$F_i = \vec{F}_i \oplus \overleftarrow{F}_i \quad \in \mathbb{R}^{B \times 215 \times 39}$$
2. Nén toàn bộ 215 khung thời gian bằng tầng `GlobalAveragePooling1D` thành vector tóm tắt $g_i$:
   $$g_i = \text{GAP1D}(F_i) \quad \in \mathbb{R}^{B \times 1 \times 39}$$
3. Ghép nối 8 vector đại diện từ 8 thang đo:
   $$\mathbf{G} = [g_1, g_2, \dots, g_8] \quad \in \mathbb{R}^{B \times 8 \times 39}$$
4. **Tầng tự định nghĩa `WeightLayer`:**
   * Tác giả không dùng trung bình cộng đơn giản, mà tạo một ma trận trọng số có thể học $\mathbf{W} \in \mathbb{R}^{8 \times 1}$.
   * Thực hiện phép nhân Dot Product để tự động điều chỉnh tầm quan trọng của từng thang đo:
     $$g_{drt} = \mathbf{G}^T \cdot \mathbf{W} \quad \in \mathbb{R}^{B \times 39}$$
   * Thang đo thời gian nào chứa nhiều thông tin cảm xúc nhất sẽ tự động được gán trọng số lớn nhất.

---

## 7. GIAI ĐOẠN 6: PHÂN LOẠI CẢM XÚC ĐẦU RA (CLASSIFIER)

Vector đại diện cuối cùng $g_{drt}$ (kích thước 39 chiều) được đưa vào bộ phân loại:
1. **Tầng Fully Connected (Dense):** Chiếu từ 39 chiều sang 8 chiều tương ứng với 8 lớp cảm xúc.
2. **Hàm Softmax:** Chuyển đổi điểm số logit thành phân phối xác suất:
   $$P(y = c \mid x) = \frac{e^{z_c}}{\sum_{j=1}^8 e^{z_j}}$$
3. **Cảm xúc dự đoán:** Là lớp có xác suất cao nhất:
   $$\hat{y} = \arg\max_{c} P(y = c \mid x)$$

### Ý nghĩa biểu đồ `Class-Discriminative` (Góc dưới bên phải)
Khi chiếu vector $g_{drt}$ xuống không gian 2D bằng t-SNE:
* 8 cụm cảm xúc (*Angry, Calm, Disgust, Fear, Happy, Neutral, Sad, Surprise*) đã được **gom cụm tách biệt hoàn toàn thành từng ốc đảo riêng biệt**.
* Chứng minh mô hình đã học thành công các biểu diễn đặc trưng phân biệt cảm xúc rõ ràng.

---

## 8. CHIẾN LƯỢC HUẤN LUYỆN & ĐÁNH GIÁ (TRAINING STRATEGY)

1. **Hàm mất mát (Loss Function):** Sử dụng Categorical Cross-Entropy kết hợp **Label Smoothing (factor = 0.1)**.
   $$y_{\text{smooth}} = y \cdot (1 - \alpha) + \frac{\alpha}{K} \quad (\alpha = 0.1, K = 8)$$
   *Tác dụng:* Ngăn chặn mô hình trở nên quá tự tin vào một nhãn duy nhất, giảm thiểu hiện tượng quá khớp (overfitting).
2. **Thuật toán tối ưu (Optimizer):** Adam với tốc độ học $\text{lr} = 0.001$, $\beta_1 = 0.93$, $\beta_2 = 0.98$, $\epsilon = 10^{-8}$.
3. **Phương pháp kiểm thử (Cross-Validation):** Đánh giá theo **10-fold Cross-Validation** (chia dữ liệu thành 10 phần, lần lượt dùng 9 phần để train và 1 phần để test độc lập, lấy trung bình cộng).

---

## 9. BẢNG TÓM TẮT KÍCH THƯỚC TENSOR (SHAPE TRACKING) QUA TỪNG TẦNG

Giả sử batch size $B = 64$ trên tập dữ liệu **RAVDESS**:

| Thứ tự | Tên tầng / Thao tác | Kích thước Tensor đầu ra (Output Shape) | Ý nghĩa |
| :---: | :--- | :---: | :--- |
| **0** | Audio Waveform gốc | $(64, 110000)$ | Sóng âm 5 giây ($fs = 22050$) |
| **1** | Sau Input Pipeline (MFCC) | $(64, 215, 39)$ | 215 khung thời gian, 39 đặc trưng |
| **2** | Nhánh Xuôi ($1\times 1$ Conv) | $(64, 215, 39)$ | Chuẩn hóa kênh luồng xuôi |
| **3** | Nhánh Ngược (Reverse + $1\times 1$ Conv) | $(64, 215, 39)$ | Đảo ngược thời gian luồng ngược |
| **4** | Khối TAB thứ $i$ ($d = 1 \to 128$) | $(64, 215, 39)$ | Trích xuất đặc trưng thời gian mở rộng |
| **5** | LIF Spiking Gating | $(64, 215, 39)$ | Xung nhị phân đóng/mở cổng $\{0, 1\}$ |
| **6** | Đầu ra TAB: $F = \vec{F}_{n-1} \odot S$ | $(64, 215, 39)$ | Đặc trưng đã lọc sạch khoảng lặng |
| **7** | Cộng hai luồng: $\vec{F}_i \oplus \overleftarrow{F}_i$ | $(64, 215, 39)$ | Tích hợp thông tin xuôi và ngược |
| **8** | Global Average Pooling 1D | $(64, 1, 39)$ | Nén 215 khung thành 1 vector đại diện |
| **9** | Gom 8 thang đo (Concatenate) | $(64, 8, 39)$ | Ma trận 8 thang đo thời gian |
| **10** | WeightLayer Dot Product | $(64, 39)$ | Vector ngữ cảnh toàn cục tối ưu $g_{drt}$ |
| **11** | Dense + Softmax | $(64, 8)$ | Xác suất của 8 lớp cảm xúc đầu ra |

---

*Tài liệu được tổng hợp chi tiết nhằm phục vụ công tác nghiên cứu, thực nghiệm và bảo vệ đồ án của tác giả ThanhBinngoc.*
