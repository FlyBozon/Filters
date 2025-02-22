import cv2
import numpy as np
import os
import ffmpeg

#added in this way, as it is easier for testing
INPUT_VIDEO = "merlin.mp4"
OUTPUT_VIDEO = "merlin_cartoon.mp4"
TEMP_VIDEO = "temp_video_no_audio.mp4"
DISPLAY_VIDEO = False  # Set to True to enable real-time display


def cartoonize_frame(frame):
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

    return cartoon


def process_video():
    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        print(f"Error: Cannot open {INPUT_VIDEO}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(TEMP_VIDEO, fourcc, frame_rate, (frame_width, frame_height))

    print("Processing video, please wait...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        cartoon_frame = cartoonize_frame(frame)
        out.write(cartoon_frame)

        if DISPLAY_VIDEO:
            cv2.imshow("Cartoon Video", cartoon_frame)  # Display video (Disabled by default)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Processing completed. Now extracting and merging audio...")


def extract_audio():
    audio_file = "extracted_audio.aac"
    ffmpeg.input(INPUT_VIDEO).output(audio_file, codec="aac", ac=2, ar="44100").run(overwrite_output=True)
    return audio_file


def merge_audio(audio_file):
    ffmpeg.concat(
        ffmpeg.input(TEMP_VIDEO),
        ffmpeg.input(audio_file),
        v=1, a=1
    ).output(OUTPUT_VIDEO, vcodec="libx264", acodec="aac").run(overwrite_output=True)

    os.remove(TEMP_VIDEO)
    os.remove(audio_file)
    print("Final video saved as", OUTPUT_VIDEO)

if __name__ == "__main__":
    process_video()
    audio_path = extract_audio()
    merge_audio(audio_path)
