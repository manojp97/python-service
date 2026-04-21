import cv2
import pytesseract
import numpy as np

def detect_number_plate(image_bytes):
    try:
        # decode image
        img = cv2.imdecode(
            np.frombuffer(image_bytes, np.uint8),
            cv2.IMREAD_COLOR
        )

        if img is None:
            return "INVALID IMAGE"

        # 🔥 Better resize (IMPORTANT)
        img = cv2.resize(img, None, fx=2, fy=2)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # smooth
        gray = cv2.bilateralFilter(gray, 11, 17, 17)

        # edge detect
        edged = cv2.Canny(gray, 30, 200)

        # find contours
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

                # 🔥 better filtering
                if 2 < ratio < 6 and w > 100:
                    # SAFE crop (IMPORTANT)
                    y1 = max(y, 0)
                    y2 = min(y + h, gray.shape[0])
                    x1 = max(x, 0)
                    x2 = min(x + w, gray.shape[1])

                    plate_img = gray[y1:y2, x1:x2]
                    break

        # ❌ no plate detected
        if plate_img is None:
            return "NO PLATE DETECTED"

        # 🔥 extra preprocess (NEW)
        plate_img = cv2.GaussianBlur(plate_img, (5, 5), 0)

        _, thresh = cv2.threshold(
            plate_img,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # 🔥 OCR
        text = pytesseract.image_to_string(
            thresh,
            config='--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        )

        # clean text
        clean_text = "".join(e for e in text if e.isalnum())

        print("OCR TEXT:", clean_text)

        if len(clean_text) < 5:
            return "TRY CLEAR IMAGE"

        return clean_text

    except Exception as e:
        return str(e)