import cv2
import numpy as np
import re
from imutils.perspective import four_point_transform
from paddleocr import PaddleOCR
import datetime

ocr_model = PaddleOCR(use_angle_cls=True, lang="en")

count = 0
scale = 0.5
WIDTH, HEIGHT = 1920, 1080

def image_processing(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Adjust contrast and brightness
    alpha = 1.5  # Contrast control (1.0-3.0)
    beta = 19  # Brightness control (0-100)
    adjusted = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)

    # Contrast adjustment with CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(adjusted)

    # Sharpening kernel
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)

    return sharpened


def parse_ocr_text(text):
      tanggal = re.search(r'(\d{2}/\d{2})\d{2}:\d{2}:\d{2}', text)
    rekening = re.search(r'Ke (\d*)', text)
    nama = re.search(r'Ke \d* (.*?) Rp', text)
    jumlah_uang = re.search(r'Rp,([\d,.]*)', text)

     if not all([tanggal, nama, rekening, jumlah_uang]):
        tanggal = re.search(r'(\d{2}/\d{2}/\d{2})', text)
        nama = re.search(r'NAMA : (.*?) JUMLAH', text)
        rekening = re.search(r'TRANSFER KE REK\. (\d*) NAMA', text)
        jumlah_uang = re.search(r'JUMLAH RP ([\d,.]*)$', text)

    return {
        'tanggal': tanggal.group(1) if tanggal else None,
        'nama': nama.group(1).strip() if nama else None,
        'rekening': rekening.group(1) if rekening else None,
        'jumlah_uang': jumlah_uang.group(1) if jumlah_uang else None,
    }


def scan_detection(image):
    document_contour = np.array([[0, 0], [WIDTH, 0], [WIDTH, HEIGHT], [0, HEIGHT]])
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, threshold = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    max_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.015 * peri, True)
            if area > max_area and len(approx) == 4:
                document_contour = approx
                max_area = area

    return document_contour

def process_ocr(filepath):
    global count  
    img = cv2.imread(filepath)
     img = cv2.resize(img, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)
    document_contour = scan_detection(img)

    processed = image_processing(img)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    output_filepath = f"output/scanned_{timestamp}_{count}.jpg"
    cv2.imwrite(output_filepath, processed)
    count += 1

    result = ocr_model.ocr(processed)
    lines = [line_info[-1][0] for line_info in result]
    text = ' '.join(lines)

    parsed_text = parse_ocr_text(text)

    return {'text': text, 'parsed_text': parsed_text, 'url': f"http://yourserver.com/files/{output_filepath}"}
