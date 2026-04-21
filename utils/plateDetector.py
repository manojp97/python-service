import cv2
import pytesseract
import numpy as np

def detect_number_plate(image_bytes):
    try:
        img = cv2.imdecode(
            np.frombuffer(image_bytes, np.uint8),
            cv2.IMREAD_COLOR
        )

        img = cv2.resize(img, None, fx=1.5, fy=1.5)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 11, 17, 17)

        edged = cv2.Canny(gray, 30, 200)

        contours, _ = cv2.findContours(
            edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

        plate_img = None

        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, 10, True)

            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(cnt)
                ratio = w / float(h)

                if 2 < ratio < 6:
                    plate_img = gray[y:y+h, x:x+w]
                    break

        if plate_img is None:
            return "PLEASE UPLOAD CLEAR PLATE IMAGE"

        _, thresh = cv2.threshold(
            plate_img, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        text = pytesseract.image_to_string(
            thresh,
            config='--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        )

        clean_text = "".join(e for e in text if e.isalnum())

        if len(clean_text) < 5:
            return "TRY CLEAR IMAGE"

        return clean_text

    except Exception as e:
        return str(e)