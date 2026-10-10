"""
gui/app.py
Giao diện phần mềm Chuyển ảnh thành tranh vẽ (Chương 5 - Xử lý ảnh số)
Xây dựng bằng Tkinter và Pillow, thuật toán thuần túy NumPy (KHÔNG sử dụng OpenCV).

Thiết kế tối giản và hiện đại:
- Khởi động sạch: Không nạp sẵn ảnh test, hiển thị nút/khung "Tải ảnh lên"
- Không cần kéo thanh trượt (Đã tối ưu hóa tự động toàn bộ trọng số tốt nhất)
- Bỏ phần chọn ảnh mẫu nghiệm thu và các tab trung gian
- Hiển thị song song trực quan: Ảnh gốc đầu vào vs. Tranh vẽ kết quả
"""

import sys
import time
import threading
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from core.sketch import process_optimal

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)


class SketchApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Phần Mềm Chuyển Ảnh Thành Tranh Vẽ Tay")
        self.geometry("1200x750")
        self.minsize(980, 640)

        # Trạng thái dữ liệu
        self.current_image_path: Optional[str] = None
        self.orig_pil: Optional[Image.Image] = None
        self.orig_np: Optional[np.ndarray] = None
        self.result_np: Optional[np.ndarray] = None
        self.current_mode: str = "pencil"

        # Image references cho Tkinter để tránh bị Garbage Collector giải phóng
        self.tk_orig: Optional[ImageTk.PhotoImage] = None
        self.tk_result: Optional[ImageTk.PhotoImage] = None

        self._init_theme()
        self._build_ui()

    def _init_theme(self):
        self.style = ttk.Style(self)
        try:
            self.style.theme_use("clam")
        except Exception:
            pass

        # Màu sắc chủ đạo: Dark Theme thanh lịch
        self.configure(bg="#18181f")
        self.style.configure(".", background="#18181f", foreground="#ffffff", font=("Segoe UI", 10))
        self.style.configure("TFrame", background="#18181f")
        
        self.style.configure(
            "TButton", 
            background="#2e2e3a", 
            foreground="#ffffff", 
            padding=[12, 6], 
            font=("Segoe UI", 10, "bold"),
            borderwidth=0
        )
        self.style.map(
            "TButton", 
            background=[("active", "#4361ee"), ("pressed", "#3a0ca3")], 
            foreground=[("active", "#ffffff")]
        )
        
        self.style.configure(
            "Accent.TButton", 
            background="#4361ee", 
            foreground="#ffffff", 
            padding=[16, 7], 
            font=("Segoe UI", 10, "bold"),
            borderwidth=0
        )
        self.style.map(
            "Accent.TButton", 
            background=[("active", "#4cc9f0"), ("pressed", "#3a0ca3")], 
            foreground=[("active", "#000000")]
        )

        self.style.configure(
            "Success.TButton", 
            background="#2ec4b6", 
            foreground="#011627", 
            padding=[14, 7], 
            font=("Segoe UI", 10, "bold"),
            borderwidth=0
        )
        self.style.map(
            "Success.TButton", 
            background=[("active", "#20a498"), ("pressed", "#0f7168")], 
            foreground=[("active", "#ffffff")]
        )

    def _build_ui(self):
        # 1. Header Toolbar
        top_bar = tk.Frame(self, bg="#20202a", padx=20, pady=12)
        top_bar.pack(fill=tk.X, side=tk.TOP)

        # Tiêu đề bên trái
        title_box = tk.Frame(top_bar, bg="#20202a")
        title_box.pack(side=tk.LEFT)
        
        lbl_title = tk.Label(
            title_box, 
            text="CHUYỂN ẢNH THÀNH TRANH VẼ TAY", 
            bg="#20202a", 
            fg="#4cc9f0", 
            font=("Segoe UI", 13, "bold")
        )
        lbl_title.pack(anchor=tk.W)

        # lbl_sub = tk.Label(
        #     title_box, 
        #     # text="Chương 5: Lọc không gian • Phát hiện biên Sobel • Thuần túy NumPy (Không OpenCV)", 
        #     bg="#20202a", 
        #     fg="#8e9aaf", 
        #     font=("Segoe UI", 9)
        # )
        # lbl_sub.pack(anchor=tk.W)

        # Các nút hành động bên phải
        action_box = tk.Frame(top_bar, bg="#20202a")
        action_box.pack(side=tk.RIGHT)

        btn_open = ttk.Button(
            action_box, 
            text="📂 TẢI ẢNH LÊN", 
            style="Accent.TButton", 
            command=self.open_image_dialog
        )
        btn_open.pack(side=tk.LEFT, padx=6)

        btn_save = ttk.Button(
            action_box, 
            text="💾 LƯU TRANH VẼ", 
            style="Success.TButton", 
            command=self.save_result_dialog
        )
        btn_save.pack(side=tk.LEFT, padx=6)

        # 2. Thanh chọn phong cách vẽ tranh (Tối ưu tự động)
        style_bar = tk.Frame(self, bg="#242430", padx=20, pady=8)
        style_bar.pack(fill=tk.X, side=tk.TOP)

        lbl_choose = tk.Label(
            style_bar, 
            text="Chọn phong cách vẽ:", 
            bg="#242430", 
            fg="#e0e0e0", 
            font=("Segoe UI", 10, "bold")
        )
        lbl_choose.pack(side=tk.LEFT, padx=(0, 15))

        self.mode_var = tk.StringVar(value="pencil")
        modes = [
            ("✏️ Tranh Chì Đen Trắng", "pencil"),
            ("🎨 Tranh Chì Màu Nghệ Thuật", "color"),
            ("🛡️ Bảo Toàn Biên Bilateral (Y tế/Công nghiệp)", "bilateral"),
            ("📐 Bản Vẽ Nét Kỹ Thuật", "technical")
        ]

        for text, val in modes:
            rb = tk.Radiobutton(
                style_bar, 
                text=text, 
                variable=self.mode_var, 
                value=val,
                bg="#242430", 
                fg="#ffffff", 
                selectcolor="#4361ee",
                activebackground="#242430", 
                activeforeground="#4cc9f0",
                font=("Segoe UI", 10),
                command=self.on_mode_change
            )
            rb.pack(side=tk.LEFT, padx=10)

        # 3. Khu vực hiển thị so sánh song song (Ảnh Gốc vs. Tranh Vẽ)
        content_frame = tk.Frame(self, bg="#18181f", padx=15, pady=12)
        content_frame.pack(fill=tk.BOTH, expand=True)

        content_frame.columnconfigure(0, weight=1, uniform="equal")
        content_frame.columnconfigure(1, weight=1, uniform="equal")
        content_frame.rowconfigure(0, weight=1)

        # --- Khung Trái: Ảnh Gốc ---
        self.box_left = tk.Frame(content_frame, bg="#20202a", relief=tk.FLAT, bd=1)
        self.box_left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))

        lbl_header_left = tk.Label(
            self.box_left, 
            text="ẢNH GỐC ĐẦU VÀO", 
            bg="#262634", 
            fg="#a0a0b0", 
            font=("Segoe UI", 10, "bold"),
            pady=6
        )
        lbl_header_left.pack(fill=tk.X, side=tk.TOP)

        self.canvas_left = tk.Label(
            self.box_left, 
            text="Chưa có ảnh được chọn\n\nVui lòng bấm '📂 TẢI ẢNH LÊN' ở góc trên để bắt đầu",
            bg="#16161f", 
            fg="#6c757d", 
            font=("Segoe UI", 11)
        )
        self.canvas_left.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # --- Khung Phải: Tranh Vẽ Kết Quả ---
        self.box_right = tk.Frame(content_frame, bg="#20202a", relief=tk.FLAT, bd=1)
        self.box_right.grid(row=0, column=1, sticky="nsew", padx=(8, 0))

        lbl_header_right = tk.Label(
            self.box_right, 
            text="TRANH VẼ KẾT QUẢ", 
            bg="#262634", 
            fg="#4cc9f0", 
            font=("Segoe UI", 10, "bold"),
            pady=6
        )
        lbl_header_right.pack(fill=tk.X, side=tk.TOP)

        self.canvas_right = tk.Label(
            self.box_right, 
            text="Tranh vẽ kết quả sẽ hiển thị tại đây", 
            bg="#16161f", 
            fg="#6c757d", 
            font=("Segoe UI", 11)
        )
        self.canvas_right.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # Xử lý sự kiện thay đổi kích thước cửa sổ để co giãn ảnh mượt mà
        self.bind("<Configure>", self._on_window_resize)

        # 4. Status Bar
        self.status_bar = tk.Label(
            self, 
            text="Sẵn sàng. Chưa có ảnh nạp.", 
            bg="#121217", 
            fg="#8e9aaf", 
            anchor=tk.W, 
            padx=16, 
            pady=6,
            font=("Segoe UI", 9)
        )
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM)

    def _on_window_resize(self, event=None):
        if self.orig_np is not None and self.result_np is not None:
            # Co giãn ảnh theo khung mới
            self._display_images()

    def open_image_dialog(self):
        filetypes = [
            ("All Images", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff;*.webp"),
            ("PNG Files", "*.png"),
            ("JPEG Files", "*.jpg;*.jpeg"),
            ("All Files", "*.*")
        ]
        path = filedialog.askopenfilename(title="Chọn ảnh đầu vào", filetypes=filetypes)
        if path:
            self.load_image(path)

    def load_image(self, path: str):
        try:
            self.current_image_path = path
            pil_img = Image.open(path)
            
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")

            self.orig_pil = pil_img
            self.orig_np = np.array(pil_img, dtype=np.uint8)

            filename = Path(path).name
            h, w = self.orig_np.shape[:2]
            self.status_bar.config(text=f"Đang xử lý: {filename} ({w}x{h} px)...")
            self.update_idletasks()

            # Chạy xử lý với phong cách đang chọn
            self.process_current_image()

        except Exception as e:
            messagebox.showerror("Lỗi nạp ảnh", f"Không thể đọc file ảnh: {e}")

    def on_mode_change(self):
        self.current_mode = self.mode_var.get()
        if self.orig_np is not None:
            self.process_current_image()

    def process_current_image(self):
        if self.orig_np is None:
            return

        self.status_bar.config(text=f"Đang tạo tranh vẽ ({self.current_mode})...")
        self.update_idletasks()

        threading.Thread(target=self._run_worker, daemon=True).start()

    def _run_worker(self):
        t0 = time.time()
        try:
            mode = self.current_mode
            result = process_optimal(self.orig_np, mode=mode)
            elapsed = time.time() - t0

            # Cập nhật kết quả vào luồng giao diện chính
            self.after(0, self._on_process_complete, result, elapsed)

        except Exception as e:
            self.after(0, lambda: self.status_bar.config(text=f"Lỗi xử lý: {e}"))

    def _on_process_complete(self, result: np.ndarray, elapsed: float):
        self.result_np = result
        self._display_images()

        mode_name_map = {
            "pencil": "Tranh chì đen trắng",
            "color": "Tranh chì màu",
            "bilateral": "Bảo toàn biên Bilateral",
            "technical": "Nét vẽ kỹ thuật"
        }
        mode_str = mode_name_map.get(self.current_mode, self.current_mode)
        filename = Path(self.current_image_path).name if self.current_image_path else ""
        self.status_bar.config(
            text=f"Hoàn thành trong {elapsed:.3f}s | File: {filename} | Phong cách: {mode_str}"
        )

    def _display_images(self):
        if self.orig_np is None:
            return

        # Kích thước khung hiển thị
        w_left = max(100, self.canvas_left.winfo_width())
        h_left = max(100, self.canvas_left.winfo_height())

        # 1. Hiển thị Ảnh Gốc bên trái
        img_orig_pil = Image.fromarray(self.orig_np)
        scale_orig = min(w_left / img_orig_pil.width, h_left / img_orig_pil.height, 1.0)
        tw_orig = max(1, int(img_orig_pil.width * scale_orig))
        th_orig = max(1, int(img_orig_pil.height * scale_orig))
        resized_orig = img_orig_pil.resize((tw_orig, th_orig), Image.Resampling.BILINEAR)

        self.tk_orig = ImageTk.PhotoImage(resized_orig)
        self.canvas_left.configure(image=self.tk_orig, text="")

        # 2. Hiển thị Tranh Vẽ bên phải
        if self.result_np is not None:
            w_right = max(100, self.canvas_right.winfo_width())
            h_right = max(100, self.canvas_right.winfo_height())

            img_res_pil = Image.fromarray(self.result_np)
            scale_res = min(w_right / img_res_pil.width, h_right / img_res_pil.height, 1.0)
            tw_res = max(1, int(img_res_pil.width * scale_res))
            th_res = max(1, int(img_res_pil.height * scale_res))
            resized_res = img_res_pil.resize((tw_res, th_res), Image.Resampling.BILINEAR)

            self.tk_result = ImageTk.PhotoImage(resized_res)
            self.canvas_right.configure(image=self.tk_result, text="")

    def save_result_dialog(self):
        if self.result_np is None:
            messagebox.showwarning("Cảnh báo", "Vui lòng tải ảnh lên trước khi lưu kết quả!")
            return

        filetypes = [
            ("PNG Image", "*.png"),
            ("JPEG Image", "*.jpg"),
            ("All Files", "*.*")
        ]
        default_name = f"sketch_{self.current_mode}.png"
        if self.current_image_path:
            stem = Path(self.current_image_path).stem
            default_name = f"{stem}_sketch_{self.current_mode}.png"

        save_path = filedialog.asksaveasfilename(
            title="Lưu ảnh tranh vẽ",
            initialfile=default_name,
            defaultextension=".png",
            filetypes=filetypes
        )
        if not save_path:
            return

        try:
            Image.fromarray(self.result_np).save(save_path)
            messagebox.showinfo("Thành công", f"Đã lưu tranh vẽ thành công tại:\n{save_path}")
            self.status_bar.config(text=f"Đã lưu kết quả tại: {Path(save_path).name}")

        except Exception as e:
            messagebox.showerror("Lỗi lưu ảnh", f"Không thể lưu ảnh: {e}")


def run_gui():
    app = SketchApp()
    app.mainloop()


if __name__ == "__main__":
    run_gui()
