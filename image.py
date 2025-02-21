import cv2
import numpy as np
import os
import time

INPUT_IMG = "sample.jpg"
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)


def save_img(img, path, title):
    fragment_path = os.path.join(path, f"{title}.png")
    cv2.imwrite(fragment_path, img)


def cartoonize_frame(frame):
    start_time = time.time()  # Record start time

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.medianBlur(gray, 7)
    edges = cv2.adaptiveThreshold(gray_blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, 9, 9)
    color = cv2.bilateralFilter(frame, d=9, sigmaColor=300, sigmaSpace=300)

    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = cv2.add(hsv[:, :, 1], 80)
    hsv[:, :, 2] = cv2.add(hsv[:, :, 2], 50)
    vivid_color = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(vivid_color, -1, kernel)
    cartoon = cv2.bitwise_and(sharpened, sharpened, mask=edges)

    end_time = time.time()
    print(f"Cartoonize frame execution time: {end_time - start_time:.4f} seconds")

    return cartoon

if __name__ == "__main__":
    image = cv2.imread(INPUT_IMG)

    if image is None:
        print(f"Error: Could not read image '{INPUT_IMG}'. Please check the file path.")
    else:
        cartoonized_img = cartoonize_frame(image)
        save_img(cartoonized_img, output_dir, "output")
        print("Image processing completed. Saved to output directory.")
