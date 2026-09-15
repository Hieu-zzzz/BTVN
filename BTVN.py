import cv2
import matplotlib.pyplot as plt
import numpy as np

# 1. NẠP HOẶC TẠO ẢNH GIẢ LẬP ĐỂ CHẠY MINH HỌA
# Thay 'input.jpg' bằng đường dẫn ảnh của bạn nếu có
img_gray = cv2.imread("io.png", cv2.IMREAD_GRAYSCALE)
img_color = cv2.imread("io.png")

if img_gray is None:
    # Tạo ảnh xám thử nghiệm có hình khối & nhiễu
    img_gray = np.zeros((300, 300), dtype=np.uint8)
    cv2.circle(img_gray, (150, 150), 80, 200, -1)
    cv2.rectangle(img_gray, (40, 40), (110, 110), 100, -1)

    # Thêm nhiễu đốm để minh họa lọc Median
    noise = np.random.randint(0, 100, img_gray.shape)
    img_gray[noise == 0] = 0
    img_gray[noise == 99] = 255

    img_color = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    cv2.circle(img_color, (150, 150), 80, (0, 0, 255), -1)  # Khối màu đỏ

L = 256  # Mức xám tối đa

# =========================================================================
# KHAI BÁO DANH SÁCH THÔNG TIN CÁC PHƯƠNG PHÁP THEO YÊU CẦU ĐỀ BÀI
# =========================================================================

methods_info = []

# -------------------------------------------------------------------------
# CỦA NHÓM 1: XỬ LÝ THEO PIXEL (POINT PROCESSING)
# -------------------------------------------------------------------------
# Negative
img_neg = (L - 1) - img_gray
methods_info.append(
    (
        "Negative (Âm bản)",
        "Ảnh xám gốc [0, L-1]",
        "s = (L - 1) - r",
        "Ảnh đảo ngược giá trị mức xám",
        "Làm nổi bật chi tiết sáng trên nền tối (ảnh X-quang, y tế)",
        img_neg,
        "gray",
    )
)

# Log Transform
c_log = 255 / np.log(1 + np.max(img_gray))
img_log = np.array(c_log * (np.log(img_gray + 1.0)), dtype=np.uint8)
methods_info.append(
    (
        "Log Transformation",
        "Ảnh có dải giá trị hẹp",
        "s = c * log(1 + r)",
        "Ảnh đã giãn mức xám vùng tối",
        "Sáng hóa vùng tối mà không làm cháy vùng sáng (Phổ Fourier)",
        img_log,
        "gray",
    )
)

# Gamma Correction
gamma = 0.5
img_gamma = np.array(255 * (img_gray / 255.0) ** gamma, dtype=np.uint8)
methods_info.append(
    (
        "Gamma Correction",
        "Ảnh bị thừa/thiếu sáng",
        f"s = c * r^{gamma}",
        "Ảnh đã hiệu chỉnh độ sáng",
        "Cân bằng độ sáng hiển thị theo đặc tính màn hình",
        img_gamma,
        "gray",
    )
)

# Contrast Stretching
r_min, r_max = np.min(img_gray), np.max(img_gray)
img_contrast = np.array(
    (img_gray - r_min) * (255.0 / (r_max - r_min + 1e-5)), dtype=np.uint8
)
methods_info.append(
    (
        "Contrast Stretching",
        "Ảnh độ tương phản thấp",
        "Kéo giãn [r_min, r_max] ra [0, L-1]",
        "Ảnh phân bổ đầy đủ dải xám",
        "Tăng độ nét cho ảnh mờ, chụp thiếu sáng",
        img_contrast,
        "gray",
    )
)

# Thresholding
_, img_thresh = cv2.threshold(img_gray, 120, 255, cv2.THRESH_BINARY)
methods_info.append(
    (
        "Thresholding (Phân ngưỡng)",
        "Ảnh xám",
        "s = 1 nếu r >= T, ngược lại s = 0",
        "Ảnh nhị phân (Binary: 0/1)",
        "Tách vật thể khỏi nền (Segmentation)",
        img_thresh,
        "gray",
    )
)

# -------------------------------------------------------------------------
# CỦA NHÓM 2: XỬ LÝ THEO VÙNG LÂN CẬN (NEIGHBORHOOD FILTERING)
# -------------------------------------------------------------------------
# Averaging
img_avg = cv2.blur(img_gray, (5, 5))
methods_info.append(
    (
        "Averaging Filter",
        "Ảnh bị nhiễu hạt",
        "Nhân chập Kernel 1/N * [1]",
        "Ảnh mượt/mờ hơn",
        "Làm mịn ảnh, giảm nhiễu Gaussian",
        img_avg,
        "gray",
    )
)

# Median
img_median = cv2.medianBlur(img_gray, 5)
methods_info.append(
    (
        "Median Filter",
        "Ảnh nhiễu muối tiêu",
        "Gán giá trị trung vị của cửa sổ",
        "Ảnh sạch nhiễu đốm",
        "Khử nhiễu đốm đen/trắng mà giữ nguyên đường biên",
        img_median,
        "gray",
    )
)

# Laplacian
img_lap = cv2.convertScaleAbs(cv2.Laplacian(img_gray, cv2.CV_64F))
methods_info.append(
    (
        "Laplacian Filter",
        "Ảnh xám gốc",
        "Đạo hàm bậc 2 không gian",
        "Ảnh chi tiết biên sắc nhọn",
        "Trích xuất biên và làm sắc nét ảnh (Sharpening)",
        img_lap,
        "gray",
    )
)

# Unsharp Masking
blurred = cv2.GaussianBlur(img_gray, (9, 9), 10.0)
img_unsharp = cv2.addWeighted(img_gray, 1.5, blurred, -0.5, 0)
methods_info.append(
    (
        "Unsharp Masking",
        "Ảnh bị mờ chi tiết",
        "g(x,y) = f(x,y) + k * g_mask",
        "Ảnh tăng cường độ sắc cạnh",
        "Tăng độ nét vùng biên trong nhiếp ảnh/in ấn",
        img_unsharp,
        "gray",
    )
)

# Roberts
k_rx = np.array([[1, 0], [0, -1]], dtype=np.float32)
k_ry = np.array([[0, 1], [-1, 0]], dtype=np.float32)
img_roberts = cv2.convertScaleAbs(
    cv2.filter2D(img_gray, -1, k_rx) + cv2.filter2D(img_gray, -1, k_ry)
)
methods_info.append(
    (
        "Roberts Operator",
        "Ảnh xám",
        "Đạo hàm bậc 1 chéo 2x2",
        "Ảnh biên chéo",
        "Phát hiện biên đơn giản, tốc độ nhanh",
        img_roberts,
        "gray",
    )
)

# Sobel
sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
img_sobel = cv2.convertScaleAbs(cv2.magnitude(sobelx, sobely))
methods_info.append(
    (
        "Sobel Edge Detection",
        "Ảnh xám",
        "Đạo hàm bậc 1 không gian 3x3",
        "Ảnh cường độ biên Gradient",
        "Tách biên vật thể theo hướng x và y",
        img_sobel,
        "gray",
    )
)

# -------------------------------------------------------------------------
# CỦA NHÓM 3: XỬ LÝ MÀU SẮC (COLOR PROCESSING)
# -------------------------------------------------------------------------
# RGB / HSI / CMY
img_rgb = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
img_hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)
img_cmy = 255 - img_rgb

methods_info.append(
    (
        "Chuyển đổi RGB / CMY / HSI",
        "Ảnh gốc không gian RGB",
        "Biến đổi ma trận không gian màu",
        "Ảnh ở không gian màu HSI/CMY",
        "Tách độ sáng và màu sắc để xử lý độc lập",
        img_hsv,
        "rgb",
    )
)

# Pseudocolor
img_pseudo = cv2.applyColorMap(img_gray, cv2.COLORMAP_JET)
methods_info.append(
    (
        "Pseudocolor (Màu giả)",
        "Ảnh xám/ảnh nhiệt",
        "Ánh xạ mức xám r -> Vector RGB",
        "Ảnh màu giả cường điệu",
        "Tăng khả năng quan sát chi tiết cho mắt người",
        cv2.cvtColor(img_pseudo, cv2.COLOR_BGR2RGB),
        "rgb",
    )
)

# Color Slicing
lower_red = np.array([0, 100, 100])
upper_red = np.array([10, 255, 255])
mask = cv2.inRange(img_hsv, lower_red, upper_red)
img_slicing = cv2.bitwise_and(img_color, img_color, mask=mask)
methods_info.append(
    (
        "Color Slicing (Cắt màu)",
        "Ảnh màu RGB/HSV",
        "Lọc dải màu [V_min, V_max]",
        "Ảnh đã cô lập vùng màu",
        "Nhận diện và phân đoạn vật thể theo màu sắc",
        cv2.cvtColor(img_slicing, cv2.COLOR_BGR2RGB),
        "rgb",
    )
)

# =========================================================================
# HIỂN THỊ VÀ IN RA KẾT QUẢ ĐÚNG CHUẨN CẤU TRÚC ĐỀ BÀI
# =========================================================================

# 1. In dạng Văn bản chi tiết
print("=" * 80)
print(f"{'BÁO CÁO CHI TIẾT CÁC PHƯƠNG PHÁP XỬ LÝ ẢNH':^80}")
print("=" * 80)

for idx, item in enumerate(methods_info, 1):
    name, inp, principle, out, purpose, img, c_type = item
    print(f"\n[{idx}] PHƯƠNG PHÁP: {name.upper()}")
    print(f"  • Input:             {inp}")
    print(f"  • Nguyên lý xử lý:   {principle}")
    print(f"  • Output:            {out}")
    print(f"  • Mục đích sử dụng:  {purpose}")

# 2. Xuất bảng lưới ảnh minh họa cho từng phương pháp
fig = plt.figure(figsize=(18, 22), dpi=150)
plt.subplots_adjust(hspace=0.6, wspace=0.3)

for idx, item in enumerate(methods_info, 1):
    name, inp, principle, out, purpose, img, c_type = item
    ax = plt.subplot(4, 4, idx)

    if c_type == "gray":
        plt.imshow(img, cmap="gray")
    else:
        plt.imshow(img)

    # Đặt tiêu đề hiển thị tên phương pháp và tóm tắt nguyên lý
    title_text = f"{idx}. {name}\n[In]: {inp}\n[Out]: {out}"
    plt.title(title_text, fontsize=9, pad=6, fontweight="bold", color="#1A5276")
    plt.axis("off")

plt.tight_layout()
plt.savefig("ket_qua_minh_hoa_tat_ca_pp.png", bbox_inches="tight")
plt.show()