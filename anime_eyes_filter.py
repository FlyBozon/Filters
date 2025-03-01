import cv2
import numpy as np
import dlib

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  

def apply_anime_filter(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)

    if len(faces) == 0:
        print("No face detected!")
        return image

    for face in faces:
        landmarks = predictor(gray, face)

        left_eye = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(36, 42)]
        right_eye = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(42, 48)]

        image = enlarge_eyes(image, left_eye, right_eye)

        image = smooth_skin(image, face)

        image = enhance_colors(image)

    return image


def enlarge_eyes(image, left_eye, right_eye):
    """Enlarges the eyes in an anime-like fashion."""
    for eye in [left_eye, right_eye]:
        x, y, w, h = cv2.boundingRect(np.array(eye))
        eye_region = image[y:y + h, x:x + w]
        eye_resized = cv2.resize(eye_region, (w + 15, h + 15))  
        image[y:y + h, x:x + w] = cv2.resize(eye_resized, (w, h))  
    return image


def smooth_skin(image, face):
    """Applies a smoothing filter to mimic anime-like skin."""
    (x, y, w, h) = (face.left(), face.top(), face.width(), face.height())
    face_region = image[y:y + h, x:x + w]
    blurred = cv2.bilateralFilter(face_region, d=15, sigmaColor=75, sigmaSpace=75)
    image[y:y + h, x:x + w] = blurred
    return image


def enhance_colors(image):
    """Boosts saturation and contrast to achieve a vivid anime look."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = cv2.add(hsv[:, :, 1], 40)  
    hsv[:, :, 2] = cv2.add(hsv[:, :, 2], 30) 
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


image = cv2.imread("face.jpg") 

anime_image = apply_anime_filter(image)

cv2.imshow("Anime Face", anime_image)
cv2.imwrite("anime_face.jpg", anime_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
