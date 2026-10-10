"""
core/sketch.py
Các giải thuật tạo hiệu ứng tranh vẽ tay (Chương 5 - Xử lý ảnh số)
Cài đặt thuần túy bằng NumPy và Toán học (KHÔNG sử dụng OpenCV).

Bao gồm:
- Tranh chì đen trắng nghệ thuật (bảo toàn chi tiết vùng sáng, chuyển sắc tự nhiên)
- Tranh vẽ chì màu nghệ thuật (nét chì thích ứng theo cấu trúc cục bộ, giữ màu sắc tự nhiên)
- Tranh chì bảo toàn biên Bilateral (chuyên dụng cho ảnh y tế & công nghiệp)
- Bản vẽ nét kỹ thuật / giải phẫu (Technical Contour)
"""

from typing import Dict, Optional
import numpy as np

from core.filters import (
    rgb_to_gray,
    gaussian_blur,
    sobel_edge_detection,
    bilateral_filter
)


def create_shaded_pencil_sketch(
    img_rgb: np.ndarray,
    blur_sigma: float = 1.8,
    edge_strength: float = 0.55,
    shading_depth: float = 0.50,
    dodge_kappa: float = 0.95
) -> np.ndarray:
    """
    Tạo tranh chì đen trắng nghệ thuật:
    - Bảo toàn tối đa chi tiết vùng sáng (bầu trời, mây, ánh sáng không bị cháy trắng 255).
    - Cân bằng độ tương phản 3 tầng: sáng (highlights), trung gian (midtones), tối (shadows).
    - Giữ chi tiết trong vùng tối của vật thể, không bị bết đen đặc.
    - Đường viền Sobel bão hòa mềm (Hill function) thanh thoát, tự nhiên, không bị thô dày.
    - Không sử dụng lưới sọc gạch chéo cố định.
    
    Tham số:
    - blur_sigma: Độ lệch chuẩn làm mượt tiền xử lý cho dò biên (mặc định 1.8)
    - edge_strength: Độ đậm của nét chì viền phác thảo (mặc định 0.55)
    - shading_depth: Mức độ phủ sắc độ than chì cho khối vật thể (mặc định 0.50)
    - dodge_kappa: Hệ số nén mẫu số Color Dodge giúp bảo toàn vùng sáng (mặc định 0.95)
    """
    gray = rgb_to_gray(img_rgb)
    h, w = gray.shape

    # 1. Phát hiện biên Sobel cho nét viền phác thảo
    blurred_for_edge = gaussian_blur(gray, sigma=blur_sigma)
    edges, _, _ = sobel_edge_detection(blurred_for_edge)

    # Ánh xạ phi tuyến mềm (Hàm Hill: M / (M + K)) giúp nét nhỏ thanh mảnh, nét chính rõ ràng,
    # loại bỏ hoàn toàn viền gai nhọn và hiện tượng viền dày đặc
    edge_norm = edges / (edges + 45.0)
    edge_stroke = 1.0 - edge_strength * edge_norm

    # 2. Thuật toán Color Dodge có kiểm soát (Highlight-preserving Color Dodge):
    # Hệ số dodge_kappa < 1.0 ngăn chặn mẫu số triệt tiêu với (255 - gray) ở các vùng sáng,
    # nhờ đó bảo toàn hoàn hảo chuyển sắc tinh tế của mây, bầu trời và các mặt phẳng sáng
    inv_gray = 255.0 - gray
    inv_blur = gaussian_blur(inv_gray, sigma=6.0)
    denom = np.maximum(255.0 - inv_blur * dodge_kappa, 1.0)
    dodge = np.clip((gray * 255.0) / denom, 0.0, 255.0) / 255.0

    # 3. Đánh bóng sắc độ vật thể bảo toàn biên (Bilateral Tonal Shading):
    # Lọc Bilateral giữ nguyên ranh giới hình khối và loại bỏ nhiễu hạt
    smooth_gray = bilateral_filter(gray, sigma_s=4.0, sigma_r=25.0, radius=2) / 255.0
    
    # Ánh xạ sắc độ: Vùng sáng (smooth_gray ~ 0.95) giữ ở mức 0.97 (sáng sạch như giấy nhưng không bão hòa),
    # Vùng trung gian và vùng tối có sắc độ chuyển tiếp tự nhiên
    shading = 1.0 - shading_depth * (1.0 - smooth_gray)

    # 4. Hòa trộn tổng thể: Lớp bóng khối * Vi cấu trúc Dodge * Nét viền phác thảo
    # Đặt ngưỡng sàn 0.12 để tránh bết đen vùng tối (crushed shadows)
    sketch = shading * (0.35 + 0.65 * dodge) * edge_stroke
    sketch = np.clip(sketch, 0.12, 1.0)

    # Chuyển về miền [0, 255] mà không dùng phép nhân kích sáng gây cháy ảnh
    final = sketch * 255.0
    return np.clip(np.round(final), 0, 255).astype(np.uint8)


def create_color_pencil_sketch(
    img_rgb: np.ndarray,
    color_smooth_s: float = 2.5,
    color_smooth_r: float = 22.0,
    edge_strength: float = 0.50,
    vibrance: float = 1.10,
    adaptive_stroke_strength: float = 0.22
) -> np.ndarray:
    """
    Tạo tranh vẽ chì màu nghệ thuật (Color Pencil Drawing):
    - Bảo toàn màu sắc tự nhiên và sự khác biệt màu sắc giữa các vùng của ảnh gốc.
    - Nét chì màu thay đổi theo cấu trúc cục bộ (theo hướng tiếp tuyến của đường biên).
    - Cường độ nét biến thiên theo độ lớn gradient: biến mất ở vùng trời phẳng, da mịn;
      rõ nét ở vùng có chi tiết và đường viền.
    - KHÔNG phủ lưới gạch chéo cố định lên toàn bộ ảnh.
    - Nét viền mang màu sắc tự nhiên đậm hơn của chính vật thể (đặc trưng của chì màu).
    
    Tham số:
    - color_smooth_s: Spatial sigma của bộ lọc Bilateral cho màu sắc (mặc định 2.5)
    - color_smooth_r: Range sigma của bộ lọc Bilateral cho màu sắc (mặc định 22.0)
    - edge_strength: Độ đậm nét chì viền cục bộ (mặc định 0.50)
    - vibrance: Hệ số tăng độ tươi tắn màu sắc vừa phải (mặc định 1.10)
    - adaptive_stroke_strength: Độ nổi của nét chì thích ứng cấu trúc (mặc định 0.22)
    """
    if img_rgb.ndim < 3:
        return create_shaded_pencil_sketch(img_rgb)

    gray = rgb_to_gray(img_rgb)
    h, w = gray.shape

    # 1. Tạo lớp màu sáp chì mịn màng (Bilateral Filter trên từng kênh RGB)
    # Khử nhiễu ảnh số, tạo mảng màu phẳng mịn như chất liệu sáp chì màu
    c_smooth = np.zeros_like(img_rgb, dtype=np.float32)
    for c in range(3):
        c_smooth[:, :, c] = bilateral_filter(img_rgb[:, :, c], sigma_s=color_smooth_s, sigma_r=color_smooth_r, radius=2)

    # 2. Điều chỉnh độ tươi tắn tự nhiên (Vibrancy) vừa phải, không làm biến đổi tone màu gốc
    c_gray = (0.299 * c_smooth[:, :, 0] + 0.587 * c_smooth[:, :, 1] + 0.114 * c_smooth[:, :, 2])[:, :, np.newaxis]
    c_vibrant = np.clip(c_gray + vibrance * (c_smooth - c_gray), 0.0, 255.0)

    # 3. Dò biên Sobel và tính toán hướng gradient
    blurred_g = gaussian_blur(gray, sigma=1.8)
    edges, gx, gy = sobel_edge_detection(blurred_g)

    # Chuẩn hóa biên dạng Hill mượt mà
    edge_norm = edges / (edges + 40.0)

    # 4. Nét chì màu thích ứng cấu trúc cục bộ (Structure-Adaptive Strokes):
    # Hướng gradient: theta = arctan2(gy, gx)
    theta = np.arctan2(gy, gx)
    
    # Tọa độ hình chiếu cục bộ theo hướng gradient:
    # Nét bút chì tạo ra dao động vi mô uốn lượn theo cấu trúc thực tế của vật thể
    y_idx, x_idx = np.mgrid[0:h, 0:w]
    proj = x_idx * np.cos(theta) + y_idx * np.sin(theta)
    local_stroke = (np.sin(proj * 0.8) + 1.0) * 0.5

    # CƯỜNG ĐỘ NÉT THAY ĐỔI THEO VÙNG:
    # Ở vùng trời, mảng sáng phẳng, da mịn (edges < 15): stroke_mask = 0 -> texture = 1.0 (KHÔNG CÓ NÉT GIẢ!)
    # Ở vùng có chi tiết, cành cây, gân lá, nếp vải: stroke_mask > 0 -> nét chì xuất hiện bám theo cấu trúc
    stroke_mask = np.clip((edges - 15.0) / 35.0, 0.0, 1.0)
    adaptive_texture = 1.0 - adaptive_stroke_strength * stroke_mask * (1.0 - local_stroke)

    # 5. Nét viền chì màu (Colored Edge Stroke):
    # Làm tối cục bộ theo chính sắc màu của vật thể (áo đỏ viền đỏ thẫm, biển xanh viền lam đậm)
    colored_stroke = 1.0 - edge_strength * edge_norm
    colored_stroke_3d = colored_stroke[:, :, np.newaxis]
    texture_3d = adaptive_texture[:, :, np.newaxis]

    # 6. Sắc thái Color Dodge vi mô kiểm soát vùng sáng
    inv_g = 255.0 - gray
    inv_blur = gaussian_blur(inv_g, sigma=5.0)
    denom = np.maximum(255.0 - inv_blur * 0.95, 1.0)
    dodge = np.clip((gray * 255.0) / denom, 0.0, 255.0) / 255.0
    dodge_3d = (dodge * 0.35 + 0.65)[:, :, np.newaxis]

    # 7. Hòa trộn lớp tranh chì màu: Sắc tố sáp chì * Nét viền màu * Nét thích ứng * Vi cấu trúc
    color_pencil = c_vibrant * colored_stroke_3d * texture_3d * dodge_3d

    final = np.clip(color_pencil, 0.0, 255.0)
    return np.clip(np.round(final), 0, 255).astype(np.uint8)


def create_bilateral_sketch(img_rgb: np.ndarray) -> np.ndarray:
    """
    Tranh chì bảo toàn biên Bilateral:
    Làm mịn phẳng nền đồng nhất mà giữ trọn độ sắc nét của đường ranh giới.
    Rất tối ưu cho ảnh y tế (X-quang, CT) và ảnh công nghiệp (PCB, cơ khí).
    """
    gray = rgb_to_gray(img_rgb)
    h, w = gray.shape

    # Lọc song phương bảo toàn cạnh
    smooth_bilat = bilateral_filter(gray, sigma_s=3.5, sigma_r=25.0, radius=2)

    # Biên Sobel từ ảnh đã khử nhiễu
    edges, _, _ = sobel_edge_detection(smooth_bilat)
    edge_stroke = 1.0 - np.clip(edges / 150.0, 0.0, 1.0) * 0.9

    # Color Dodge trên ảnh đảo
    inv = 255.0 - smooth_bilat
    inv_blur = bilateral_filter(inv, sigma_s=4.0, sigma_r=30.0, radius=2)
    denom = np.maximum(255.0 - inv_blur, 1.0)
    dodge = np.clip((smooth_bilat * 255.0) / denom, 0.0, 255.0) / 255.0

    # Phủ bóng nhẹ vùng mô mềm / bề mặt kim loại
    shading = 0.50 + 0.50 * (smooth_bilat / 255.0)

    sketch = dodge * shading * edge_stroke
    final = np.clip(sketch * 1.15, 0.0, 1.0) * 255.0
    return np.clip(np.round(final), 0, 255).astype(np.uint8)


def create_technical_sketch(img_rgb: np.ndarray) -> np.ndarray:
    """
    Bản phác thảo nét kỹ thuật / giải phẫu (Technical Contour):
    Đường viền mực/chì sắc nét trên nền giấy trắng sáng sạch, loại bỏ hoàn toàn nhiễu nền.
    """
    gray = rgb_to_gray(img_rgb)
    smoothed = bilateral_filter(gray, sigma_s=3.0, sigma_r=25.0, radius=2)
    edges, _, _ = sobel_edge_detection(smoothed, threshold=22.0)

    # Nét mực đen trên nền giấy trắng
    inv_edges = 255.0 - edges
    light_shading = 255.0 - (255.0 - smoothed) * 0.15
    result = (inv_edges / 255.0) * light_shading
    return np.clip(np.round(result), 0, 255).astype(np.uint8)


def process_optimal(img_rgb: np.ndarray, mode: str = "pencil") -> np.ndarray:
    """
    Tự động áp dụng bộ trọng số tối ưu nhất cho từng phong cách vẽ:
    - 'pencil': Tranh chì đen trắng (bảo toàn chi tiết vùng sáng, chuyển sắc tự nhiên)
    - 'color': Tranh vẽ chì màu nghệ thuật (nét chì thích ứng cấu trúc, màu sắc tự nhiên)
    - 'bilateral': Tranh chì bảo toàn biên (Y tế & Công nghiệp)
    - 'technical': Bản vẽ nét kỹ thuật
    """
    if mode == "color":
        return create_color_pencil_sketch(img_rgb)
    elif mode == "bilateral":
        return create_bilateral_sketch(img_rgb)
    elif mode == "technical":
        return create_technical_sketch(img_rgb)
    else: # pencil
        return create_shaded_pencil_sketch(img_rgb)
