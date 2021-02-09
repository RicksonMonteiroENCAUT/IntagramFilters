import cv2
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
mascara=cv2.imread("mascara.jpg")
scale=1


cap=cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame=cap.read()
    frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_rects = face_classifier.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in face_rects:
        if h > 0 and w > 0:
            h, w = int(1.5 * h), int(1.2 * w)
            y -= 25
            x -= 17

            frame_roi = frame[y:y + h, x:x + w]
            face_mask_small = cv2.resize(mascara, (w, h), interpolation=cv2.INTER_AREA)
            mask= cv2.cvtColor(face_mask_small, cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            masked_face = cv2.bitwise_and(face_mask_small, face_mask_small, mask=thresh)
            masked_frame = cv2.bitwise_and(frame_roi, frame_roi, mask=cv2.bitwise_not(thresh))
            frame[y:y+h,x:x+w]=cv2.add(masked_face,masked_frame)
            cv2.imshow("frame",frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
