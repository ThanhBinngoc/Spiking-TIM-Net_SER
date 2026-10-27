# TỔNG HỢP KIẾN THỨC VÀ HƯỚNG DẪN NGHIÊN CỨU DỰ ÁN
## Spiking-TIM-Net: Nhận dạng Cảm xúc Giọng nói (Speech Emotion Recognition - SER)

> Tài liệu này được biên soạn nhằm tổng hợp toàn bộ lý thuyết cốt lõi, kiến trúc mô hình, kỹ thuật cải tiến nơ-ron xung (LIF SNN) và các lưu ý thực nghiệm phục vụ việc học tập, báo cáo và bảo vệ đồ án/luận văn.

---

## MỤC LỤC
1. [Tổng quan về bài toán & Mô hình gốc TIM-Net](#1-tổng-quan-về-bài-toán--mô-hình-gốc-tim-net)
2. [Chi tiết Kiến trúc Mô hình (End-to-End)](#2-chi-tiết-kiến-trúc-mô-hình-end-to-end)
3. [Điểm mới: Cải tiến Nơ-ron xung LIF (Spiking Attention at F)](#3-điểm-mới-cải-tiến-nơ-ron-xung-lif-spiking-attention-at-f)
4. [Các chỉ số đánh giá kết quả (Evaluation Metrics)](#4-các-chỉ-số-đánh-giá-kết-quả-evaluation-metrics)
5. [Cẩm nang kỹ thuật & Xử lý lỗi khi train trên Server](#5-cẩm-nang-kỹ-thuật--xử-lý-lỗi-khi-train-trên-server)
6. [Lệnh chạy thực nghiệm đối chứng (Benchmark)](#6-lệnh-chạy-thực-nghiệm-đối-chứng-benchmark)

---

## 1. TỔNG QUAN VỀ BÀI TOÁN & MÔ HÌNH GỐC TIM-NET

### 1.1. Bài toán SER (Speech Emotion Recognition)
* **Mục tiêu:** Phân loại trạng thái cảm xúc của con người từ tín hiệu tiếng nói (ví dụ tập **RAVDESS** gồm 8 cảm xúc: *Angry, Calm, Disgust, Fear, Happy, Neutral, Sad, Surprise*).
* **Thách thức:** Cảm xúc trong lời nói phân bố ở nhiều khoảng thời gian khác nhau (từ âm vị ngắn đến cả câu nói dài) và dễ bị ảnh hưởng bởi tạp âm, khoảng lặng, tiếng thở.

### 1.2. Mô hình gốc TIM-Net (ICASSP 2023)
* **Tên gọi:** **T**emporal-aware b**I**-direction **M**ulti-scale **Net**work.
* **Đặc điểm nổi bật:**
  * **Siêu nhẹ (Lightweight):** Chỉ ~**0.40 MB** (~100.000 tham số), nhỏ hơn rất nhiều so với các mô hình trước đó (CPAC: 1.23 MB, GM-TCN: 1.13 MB).
  * **Hai chiều (Bi-direction):** Xử lý đồng thời cả luồng thời gian xuôi (Forward) và đảo ngược (Backward) để nắm bắt cả ngữ cảnh quá khứ lẫn tương lai.
  * **Đa thang đo (Multi-scale):** Sử dụng các khối tích chập giãn nở với hệ số tăng theo lũy thừa ($2^0, 2^1, \dots, 2^7$) để bao quát từ âm tiết ngắn đến toàn bộ câu nói.

---

## 2. CHI TIẾT KIẾN TRÚC MÔ HÌNH (END-TO-END)

Kiến trúc gồm 4 khối liên hoàn:

```
[Input Pipeline] ──► [2 Luồng Xuôi/Ngược] ──► [Dynamic Fusion] ──► [Classifier]
  (Speech → MFCC)         (Chuỗi khối TAB)       (Gộp đa thang đo)     (FC + Softmax)
```

### 2.1. Khối 1: Input Pipeline
* **Đầu vào âm thanh thô (`Speech`):** Sóng âm thanh dạng sóng `.wav` (tần số lấy mẫu $fs = 22.050\text{ Hz}$).
* **Cân bằng độ dài (Length Normalization):** Chuẩn hóa độ dài tín hiệu về cố định theo dataset (RAVDESS là `110.000` mẫu ~ 5 giây). Ngắn hơn thì đệm thêm số 0 ở 2 đầu (`Zero-padding`), dài hơn thì cắt đều 2 đầu.
* **Trích xuất MFCC 39 chiều:**
  * Cắt khung ngắn $50\text{ ms}$, bước trượt $12.5\text{ ms}$.
  * Nhân cửa sổ Hamming $\to$ Biến đổi Fourier (FFT) $\to$ Bộ lọc thang tần số Mel $\to$ Biến đổi Cosin rời rạc (DCT) lấy 39 hệ số đầu tiên.
* **Kích thước đầu ra:** Ma trận shape $(215, 39)$ — gồm **215 khung thời gian**, mỗi khung có **39 đặc trưng tần số**.
* **Ý nghĩa biểu đồ `Class-Confusion` (t-SNE):** Đặc trưng MFCC ban đầu bị trộn lẫn hỗn độn, chứng minh bản thân MFCC tĩnh chưa thể tự phân loại cảm xúc mà cần mạng thời gian phía sau bóc tách.

### 2.2. Khối 2: TIM-Net Core & Khối Temporal-Aware Block (TAB)
* Đầu vào được tách thành **2 nhánh**:
  * Nhánh xuôi (Forward): đi qua $1\times 1$ Conv $\to$ chuỗi khối $\vec{\mathcal{T}}_1, \dots, \vec{\mathcal{T}}_n$.
  * Nhánh ngược (Reverse Time): lật ngược trục thời gian $\to$ qua $1\times 1$ Conv $\to$ chuỗi khối $\overleftarrow{\mathcal{T}}_1, \dots, \overleftarrow{\mathcal{T}}_n$.
* **Bên trong mỗi khối TAB (Temporal-Aware Block):**
  $$\vec{F}_{n-1} \longrightarrow [\text{DC Conv} \to \text{BN} \to \text{ReLU} \to \text{SD}] \times 2 \longrightarrow \sigma \longrightarrow \odot \vec{F}_{n-1} \longrightarrow \vec{F}_n$$
  * **DC Conv (Dilated Causal Conv):** Tích chập giãn nở theo luật nhân quả. Hệ số giãn nở tăng dần $d = 2^{i-1}$ ($1, 2, 4, 8, 16, 32, 64, 128$), giúp mở rộng vùng tiếp nhận (receptive field) bao phủ toàn bộ câu nói mà không làm tăng số tham số.
  * **BN (Batch Normalization):** Chuẩn hóa lô giúp ổn định tốc độ học.
  * **ReLU:** Hàm phi tuyến tính cho mạng học đặc trưng âm học sâu.
  * **SD (Spatial Dropout 1D):** Tắt ngẫu nhiên 10% các kênh đặc trưng trên toàn bộ trục thời gian để chống học vẹt (overfitting).
  * **Cổng chú ý $\sigma$ (Sigmoid):** Tính trọng số chú ý để nhân phần tử ($\odot$) với $\vec{F}_{n-1}$ tạo ra $\vec{F}_n$.

### 2.3. Khối 3: Dynamic Fusion (Hợp nhất Đa thang đo Thời gian)
* Sau mỗi tầng thời gian thứ $i$, đặc trưng xuôi và ngược được cộng lại: $\vec{F}_i \oplus \overleftarrow{F}_i$.
* Đưa qua `GlobalAveragePooling1D` để nén thành vector $g_i$.
* Gom tất cả $n$ vector $\{g_1, g_2, \dots, g_n\}$ lại, đưa qua tầng tự định nghĩa **`WeightLayer`** để tự động học trọng số kết hợp thành vector đại diện duy nhất $g_{drt}$.

### 2.4. Khối 4: Classifier (Phân loại Cảm xúc)
* Vector $g_{drt}$ đi qua tầng `Dense` + `Softmax` xuất ra xác suất của 8 lớp cảm xúc.
* **Ý nghĩa biểu đồ `Class-Discriminative` (t-SNE):** Sau khi qua TIM-Net, 8 cụm cảm xúc đã được tách biệt rõ ràng.

---

## 3. ĐIỂM MỚI: CẢI TIẾN NƠ-RON XUNG LIF (SPIKING ATTENTION AT F)

### 3.1. Động lực nghiên cứu (Research Motivation)
* **Tính sinh học (Neuromorphic):** Tai người chuyển đổi sóng âm thành các chuỗi xung thần kinh (spikes) truyền lên não.
* **Lọc nhiễu & Dập tắt khoảng lặng (Sparsity):** Trong câu nói có nhiều khoảng lặng/tiếng thở. Bản gốc dùng `Sigmoid` mềm ($0.0 \to 1.0$) vẫn giữ lại năng lượng rác làm loãng đặc trưng. Nơ-ron xung sẽ **dập tắt hoàn toàn về 0**, chỉ bắn **xung 1** khi có cảm xúc mạnh.
* **Tiết kiệm năng lượng (Energy Efficiency):** Giảm chi phí tính toán lý thuyết nhờ độ thưa của xung.

### 3.2. Cơ chế toán học của Nơ-ron LIF (Leaky Integrate-and-Fire)
Tại vị trí cổng $F$, tín hiệu đi qua phương trình điện thế màng tế bào theo thời gian $t$:
$$V(t) = \tau \cdot V(t-1) \cdot (1 - S(t-1)) + I(t)$$
$$S(t) = \text{Heaviside}(V(t) - V_{th}) = \begin{cases} 1.0 & \text{khi } V(t) \ge V_{th} \quad (\text{Bắn xung, cổng mở}) \\ 0.0 & \text{khi } V(t) < V_{th} \quad (\text{Không xung, cổng đóng}) \end{cases}$$

* $\tau$ (Decay rate): Hệ số rò rỉ điện thế (mặc định $0.85$).
* $V_{th}$ (Threshold): Ngưỡng kích hoạt phát xung (mặc định $0.5$).
* $(1 - S(t-1))$: Cơ chế **Hard Reset** — nếu bước trước vừa bắn xung thì điện thế xả về 0.

### 3.3. Kỹ thuật Surrogate Gradient (Giải quyết bài toán đứt gãy đạo hàm)
* **Vấn đề:** Hàm bước nhảy Heaviside có đạo hàm bằng 0 khắp nơi $\to$ Backpropagation bị đứt gãy, mạng không học được.
* **Giải pháp:** Sử dụng hàm **Fast Sigmoid** làm đạo hàm đại diện khi lan truyền ngược:
  $$\frac{\partial S}{\partial V} \approx \frac{1}{(1 + \alpha |V - V_{th}|)^2} \quad (\alpha = 10.0)$$
  * *Chiều xuôi (Forward):* Vẫn bắn xung nhị phân 0 hoặc 1 chuẩn xác.
  * *Chiều ngược (Backward):* Gradient truyền qua bình thường để cập nhật trọng số.

---

## 4. CÁC CHỈ SỐ ĐÁNH GIÁ KẾT QUẢ (EVALUATION METRICS)

| Chỉ số | Ý nghĩa | Công thức |
| :--- | :--- | :--- |
| **Precision** | Trong các mẫu đoán là cảm xúc X, có bao nhiêu % đúng thật? (Đoán chắc, ít báo động giả) | $\frac{TP}{TP + FP}$ |
| **Recall** | Trong các mẫu thực tế là cảm xúc X, tìm ra được bao nhiêu %? (Không bỏ sót mẫu) | $\frac{TP}{TP + FN}$ |
| **F1-Score** | Trung bình điều hòa giữa Precision và Recall (Đo lường toàn diện) | $2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$ |
| **Support** | Tổng số file âm thanh thực tế của cảm xúc đó có trong tập test. | Đếm số mẫu |
| **Sparsity (%)** | Tỷ lệ phần trăm xung bằng 0 (Đo lường mức độ tiết kiệm năng lượng của SNN). | $\frac{\text{Số xung 0}}{\text{Tổng số xung}}$ |

---

## 5. CẨM NANG KỸ THUẬT & XỬ LÝ LỖI KHI TRAIN TRÊN SERVER

### 5.1. Nguyên tắc an toàn trên Server dùng chung
1. **Luôn dùng môi trường ảo riêng:** `source ~/snn_project1/.venv/bin/activate` (Tuyệt đối không pip install toàn cục).
2. **Kiểm tra GPU còn trống:** Gõ `nvidia-smi`, chọn GPU không có ai dùng.
3. **Cấu hình GPU an toàn:** Đảm bảo `config.gpu_options.allow_growth = True` để không chiếm dụng hết VRAM của người khác.

### 5.2. Các lỗi kinh điển và cách xử lý
* **Lỗi `UnpicklingError: pickle data was truncated`:** File `RAVDE.npy` bị tải đứt đoạn (chỉ có 1MB thay vì 47MB). Cách sửa: kéo thả lại file 47MB chuẩn lên server.
* **Lỗi `Autotuner could not compile any configs for HLO (Triton/XLA)`:** Xảy ra trên các dòng card mới như **RTX PRO 6000 Blackwell (sm_120)** do trình JIT chưa tương thích. Cách sửa:
  * Thêm `jit_compile = False` vào `self.model.compile()` trong `Model.py`.
  * Thêm `tf.config.optimizer.set_jit(False)` vào `main.py`.
  * Export biến môi trường: `export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"`.

---

## 6. LỆNH CHẠY THỰC NGHIỆM ĐỐI CHỨNG (BENCHMARK)

Di chuyển vào thư mục code:
```bash
cd ~/snn_project1/SER/TIM-Net_SER-main/Code
```

### 6.1. Chạy mô hình cải tiến LIF Spiking (Mô hình đề xuất):
```bash
nohup python main.py --mode train --data RAVDE --split_fold 10 --random_seed 46 --epoch 500 --gpu 0 --use_spike True --tau 0.85 --threshold 0.5 > train_lif_500.log 2>&1 &
```

### 6.2. Chạy mô hình Baseline gốc (Để lấy số liệu so sánh đối chứng):
```bash
nohup python main.py --mode train --data RAVDE --split_fold 10 --random_seed 46 --epoch 500 --gpu 0 --use_spike False > train_base_500.log 2>&1 &
```

### 6.3. Kiểm tra tiến trình đang chạy ngầm:
```bash
tail -f train_lif_500.log
```

---

*Tài liệu này được biên soạn bởi ThanhBinngoc phục vụ nghiên cứu và phát triển mô hình Spiking-TIM-Net.*
