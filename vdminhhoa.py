import cv2
import numpy as np
import os

# =========================================================
# 1. ĐỌC ẢNH
# =========================================================

image_path = r"OIP.jpeg"

img = cv2.imread(image_path)

if img is None:
    print("Khong doc duoc anh!")
    exit()

# Resize nếu ảnh quá lớn
img = cv2.resize(img, (700, 700))

# OpenCV đọc ảnh theo BGR
# Chuyển sang RGB để đúng không gian màu
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# =========================================================
# 2. MEDIAN FILTER – KHỬ NHIỄU
# =========================================================

median = cv2.medianBlur(img, 5)


# =========================================================
# 3. RGB → HSI
# =========================================================

# OpenCV không có hàm RGB -> HSI trực tiếp,
# nên tự xây dựng chuyển đổi RGB sang HSI.

rgb_float = rgb.astype(np.float32) / 255.0

R = rgb_float[:, :, 0]
G = rgb_float[:, :, 1]
B = rgb_float[:, :, 2]

# Intensity
I = (R + G + B) / 3.0

# Saturation
min_rgb = np.minimum(np.minimum(R, G), B)

S = np.zeros_like(I)

mask_I = I > 0
S[mask_I] = 1 - min_rgb[mask_I] / I[mask_I]

# Hue
numerator = 0.5 * ((R - G) + (R - B))
denominator = np.sqrt(
    (R - G) ** 2 +
    (R - B) * (G - B)
)

# tránh chia cho 0
denominator = np.maximum(denominator, 1e-6)

theta = np.arccos(
    np.clip(numerator / denominator, -1, 1)
)

H = theta.copy()

# Nếu B > G thì H = 2π - theta
H[B > G] = 2 * np.pi - theta[B > G]

# Đổi H từ radian sang độ
H = H * 180 / np.pi


# =========================================================
# 4. COLOR SLICING – TÁCH MÀU XANH
# =========================================================

# Khoảng Hue cho màu xanh của quả táo.
# Có thể điều chỉnh tùy ảnh và điều kiện ánh sáng.

green_low = 45
green_high = 110

green_mask = cv2.inRange(
    H.astype(np.float32),
    green_low,
    green_high
)

# ---------------------------------------------------------
# Kết hợp thêm điều kiện Saturation để loại nền trắng
# ---------------------------------------------------------

saturation_mask = (S > 0.20).astype(np.uint8) * 255

green_mask = cv2.bitwise_and(
    green_mask,
    saturation_mask
)


# =========================================================
# 5. THRESHOLDING
# =========================================================

# Làm sạch mask trước threshold
kernel = np.ones((5, 5), np.uint8)

green_mask = cv2.morphologyEx(
    green_mask,
    cv2.MORPH_OPEN,
    kernel
)

green_mask = cv2.morphologyEx(
    green_mask,
    cv2.MORPH_CLOSE,
    kernel
)

# Threshold
_, threshold = cv2.threshold(
    green_mask,
    127,
    255,
    cv2.THRESH_BINARY
)


# =========================================================
# 6. TÌM CONTOUR – XÁC ĐỊNH QUẢ TÁO
# =========================================================

contours, _ = cv2.findContours(
    threshold,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

result = img.copy()

# Lấy contour lớn nhất
if len(contours) > 0:

    largest_contour = max(
        contours,
        key=cv2.contourArea
    )

    area = cv2.contourArea(largest_contour)

    # Chỉ nhận vật thể đủ lớn
    if area > 500:

        # Vẽ contour
        cv2.drawContours(
            result,
            [largest_contour],
            -1,
            (0, 0, 255),
            3
        )

        # Bounding box
        x, y, w, h = cv2.boundingRect(
            largest_contour
        )

        cv2.rectangle(
            result,
            (x, y),
            (x + w, y + h),
            (255, 0, 0),
            3
        )

        # Tâm vật thể
        M = cv2.moments(largest_contour)

        if M["m00"] != 0:

            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            cv2.circle(
                result,
                (cx, cy),
                7,
                (0, 255, 255),
                -1
            )

            cv2.putText(
                result,
                f"Center: ({cx}, {cy})",
                (x, y - 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

        # Hiển thị kích thước
        cv2.putText(
            result,
            f"Width: {w}px",
            (x, y - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

        cv2.putText(
            result,
            f"Height: {h}px",
            (x, y + h + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

        cv2.putText(
            result,
            f"Area: {area:.0f}px2",
            (x, y + h + 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )


# =========================================================
# 7. SOBEL – PHÁT HIỆN BIÊN
# =========================================================

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sobel_x = cv2.Sobel(
    gray,
    cv2.CV_64F,
    1,
    0,
    ksize=3
)

sobel_y = cv2.Sobel(
    gray,
    cv2.CV_64F,
    0,
    1,
    ksize=3
)

sobel = cv2.magnitude(
    sobel_x.astype(np.float32),
    sobel_y.astype(np.float32)
)

sobel = cv2.convertScaleAbs(sobel)


# =========================================================
# 8. LƯU KẾT QUẢ
# =========================================================

output_dir = "ket_qua_tao_xanh"

os.makedirs(output_dir, exist_ok=True)

cv2.imwrite(
    f"{output_dir}/01_anh_goc.png",
    img
)

cv2.imwrite(
    f"{output_dir}/02_median.png",
    median
)

cv2.imwrite(
    f"{output_dir}/03_color_slicing.png",
    green_mask
)

cv2.imwrite(
    f"{output_dir}/04_threshold.png",
    threshold
)

cv2.imwrite(
    f"{output_dir}/05_sobel.png",
    sobel
)

cv2.imwrite(
    f"{output_dir}/06_ket_qua.png",
    result
)


# =========================================================
# 9. HIỂN THỊ KẾT QUẢ
# =========================================================

cv2.imshow("1. Anh goc", img)

cv2.imshow(
    "2. Median Filter",
    median
)

cv2.imshow(
    "3. Color Slicing - Tao xanh",
    green_mask
)

cv2.imshow(
    "4. Thresholding",
    threshold
)

cv2.imshow(
    "5. Sobel",
    sobel
)

cv2.imshow(
    "6. Ket qua phat hien tao xanh",
    result
)

print("======================================")
print("PHAT HIEN TAO XANH")
print("======================================")

if len(contours) > 0:
    print(f"Dien tich: {area:.2f} pixel^2")
    print(f"Bounding Box: {w} x {h} pixel")
    print(f"Vi tri: ({x}, {y})")
else:
    print("Khong tim thay tao xanh!")

print("======================================")
print("Nhan phim bat ky de thoat...")

cv2.waitKey(0)
cv2.destroyAllWindows()