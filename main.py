"""
main.py
Đề tài: Xây dựng phần mềm chuyển ảnh thành tranh vẽ (Chương 5 - Xử lý ảnh số)
Đặc tả kỹ thuật:
- Phát hiện biên (Sobel) + Chuyển đổi mức xám + Kỹ thuật làm mịn (Gaussian Blur & Bilateral Filter).
- Đánh bóng vật thể bên trong cạnh tối hơn nền, hiệu ứng tranh chì màu chân thực.
- Giao diện trực quan: Tải ảnh lên và xem kết quả so sánh song song với các trọng số tối ưu nhất.
"""

import sys
import argparse
from pathlib import Path
from PIL import Image
import numpy as np

# Đảm bảo console Windows in tiếng Việt UTF-8 không bị lỗi charmap UnicodeEncodeError
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from core.sketch import process_optimal
from gui.app import run_gui

IMAGE_DIR = BASE_DIR / "images"
RESULT_DIR = BASE_DIR / "results"
RESULT_DIR.mkdir(exist_ok=True)


def run_batch_cli():
    """
    Chế độ dòng lệnh (CLI): Tự động xử lý các ảnh kiểm thử với bộ trọng số tối ưu nhất
    và lưu kết quả hoàn thiện vào thư mục results/
    """
    print("=" * 70)
    print("  CHƯƠNG TRÌNH XỬ LÝ ẢNH THÀNH TRANH VẼ - CHƯƠNG 5 (KHÔNG DÙNG OPENCV)")
    print("=" * 70)

    sample_files = [
        ("Tự nhiên - Chân dung", IMAGE_DIR / "ST01.png", "pencil"),
        ("Tự nhiên - Phong cảnh", IMAGE_DIR / "input.jpg", "color"),
        ("Y tế - X-quang", IMAGE_DIR / "medical" / "xray_chest.jpg", "bilateral"),
        ("Công nghiệp - Bản mạch", IMAGE_DIR / "industrial" / "pcb_board.jpg", "technical"),
    ]

    for label, path, mode in sample_files:
        if not path.exists():
            continue

        print(f"\n[+] Đang xử lý: {path.name} ({label})")
        print(f"    -> Phong cách: {mode.upper()} (Tự động áp dụng trọng số tối ưu)")

        img_pil = Image.open(path).convert("RGB")
        img_np = np.array(img_pil, dtype=np.uint8)

        res = process_optimal(img_np, mode=mode)

        out_path = RESULT_DIR / f"{path.stem}_sketch_{mode}.png"
        Image.fromarray(res).save(out_path)
        print(f"    ✔ Đã lưu kết quả hoàn thiện: results/{out_path.name}")

    print("\n" + "=" * 70)
    print("  ĐÃ XỬ LÝ XONG VÀ LƯU VÀO THƯ MỤC results/ !")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Chương trình chuyển ảnh thành tranh vẽ (Chương 5)")
    parser.add_argument("--cli", action="store_true", help="Chạy chế độ dòng lệnh tự động xuất kết quả mẫu")
    args = parser.parse_args()

    if args.cli:
        run_batch_cli()
    else:
        print("Đang khởi động Giao diện phần mềm đồ họa (GUI)...")
        run_gui()


if __name__ == "__main__":
    main()
