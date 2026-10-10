# PHẦN MỀM CHUYỂN ẢNH THÀNH TRANH VẼ

Dự án cài đặt phần mềm chuyển đổi ảnh màu thành tranh vẽ dựa trên các kỹ thuật xử lý ảnh trong miền không gian (Spatial Filtering)


---

## 1. Cấu Trúc Dự Án
```text
Photo_to_Sketch/
├── core/
│   ├── filters.py        # Tích chập tách rời 1D, Gaussian Blur, Sobel, Bilateral Filter (NumPy thuần)
│   └── sketch.py         # Color Dodge, Tranh chì kinh điển, Tranh chì màu, Technical Sketch
├── gui/
│   └── app.py            # Giao diện đồ họa Tkinter
├── images/               # Thư mục lưu trữ hình ảnh mẫu    
├── results/              # Thư mục lưu trữ kết quả
├── main.py               # Điểm khởi chạy phần mềm 
├── requirements.txt      # Thư viện yêu cầu 
└── README.md
```

---

## 2. Cơ Sở Lý Thuyết & Thuật Toán Cài Đặt

### 2.1. Chuyển đổi mức xám (Grayscale Conversion)
Sử dụng công thức chuẩn độ chói quang học ITU-R BT.601:
$$Y = 0.299 \cdot R + 0.587 \cdot G + 0.114 \cdot B$$

### 2.2. Tích chập tối ưu hóa 2D & Tách rời 1D (Separable Convolution)
Thay vì sử dụng 2 vòng lặp lồng nhau duyệt từng pixel chậm chạp ($O(H \times W)$), thuật toán khai thác tính chất tách rời của ma trận lọc Gaussian và Sobel:
$$K_{2D} = K_y \otimes K_x$$
Thực hiện tích chập theo chiều dọc với $K_y$ sau đó tích chập theo chiều ngang với $K_x$ bằng kỹ thuật cắt lát ma trận (Vectorized Slicing) trong NumPy. Tốc độ xử lý tăng hơn **1000 lần** (từ 60 giây giảm xuống dưới 0.05 giây).

### 2.3. Lọc làm mịn Gaussian (Gaussian Smoothing Filter)
Hàm trọng số Gaussian 1D:
$$G(x) = \frac{1}{\sqrt{2\pi}\sigma} \exp\left(-\frac{x^2}{2\sigma^2}\right)$$
Tự động sinh kích thước kernel theo độ lệch chuẩn: $k = 2\lceil 3\sigma \rceil + 1$.

### 2.4. Phát hiện biên Sobel (Gradient Edge Detection)
Tính đạo hàm bậc nhất theo hai hướng:
* Ma trận lọc $G_x = [1, 2, 1]^T \otimes [-1, 0, 1]$
* Ma trận lọc $G_y = [-1, 0, 1]^T \otimes [1, 2, 1]$
* Độ lớn Gradient:
$$M(x, y) = \sqrt{G_x(x, y)^2 + G_y(x, y)^2}$$

### 2.5. Lọc song phương bảo toàn cạnh (Bilateral Filter)
Khắc phục nhược điểm làm mờ biên của Gaussian filter bằng cách kết hợp đồng thời khoảng cách không gian $g_s$ và độ tương đồng mức xám $g_r$:
$$I^{\text{filtered}}(p) = \frac{1}{W_p} \sum_{q \in \Omega} I(q) \cdot \exp\left(-\frac{\|p-q\|^2}{2\sigma_s^2}\right) \cdot \exp\left(-\frac{(I(p)-I(q))^2}{2\sigma_r^2}\right)$$
* **Ý nghĩa với ảnh y tế**: Khử nhiễu mô mềm mà giữ nguyên đường ranh giới xương và tổn thương.
* **Ý nghĩa với ảnh công nghiệp**: Làm phẳng bề mặt kim loại mà giữ sắc nét đường viền mạch và linh kiện điện tử.

### 2.6. Hiệu ứng Tranh vẽ (Color Dodge Sketch & Blending)
Áp dụng công thức hòa trộn Color Dodge:
$$\text{Sketch}(x, y) = \min\left(\frac{I(x, y) \times 255}{255 - I_{\text{blur}}(x, y) + \epsilon}, 255\right)$$
Hỗ trợ đa dạng phong cách:
1. **Tranh chì đen trắng**
2. **Tranh chì màu**
3. **Tranh chì bảo toàn biên**
4. **Bản vẽ nét kỹ thuật**

---

## 3. Hướng Dẫn Sử Dụng

### 3.1. Cài đặt môi trường
```bash
pip install -r requirements.txt
```

### 3.2. Khởi chạy Giao diện Phần mềm (GUI)
Khởi động phần mềm với giao diện trực quan:
```bash
python main.py
```
* **Mở ảnh từ máy**: Hỗ trợ mọi định dạng (`.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff`).
* **Chọn ảnh mẫu nghiệm thu**: Chọn nhanh các mẫu ảnh Y tế (X-quang), Công nghiệp (PCB), Tự nhiên,...
* **Lưu kết quả**: Xuất ảnh tranh vẽ chất lượng gốc cao nhất hoặc xuất trọn bộ 4 bước vào thư mục `results/`.

### 3.3. Chạy chế độ dòng lệnh tự động (Batch CLI)
Để xử lý hàng loạt toàn bộ ảnh mẫu và lưu kết quả tự động:
```bash
python main.py --cli
```


Sau khi chạy phần mềm, chúng ta có thể nhận được kết quả sau:
<table>
  <tr>
    <th>Ảnh gốc</th>
    <th>Tranh chì đen trắng</th>
    <th>Tranh chì màu</th>
    <th>Tranh chì bảo toàn biên</th>
    
  </tr>
  <tr>
    <td align="center">
      <img src="images/ST01.png" width="250">
    </td>
    <td align="center">
      <img src="results/ST01_sketch_pencil.png" width="250">
    </td>
    <td align="center">
      <img src="results/ST01_sketch_color.png" width="250">
    </td>
    <td align="center">
      <img src="results/ST01_sketch_bilateral.png" width="250">
    </td>
  </tr>
</table>
