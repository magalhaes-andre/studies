import cv2

tracker = cv2.legacy.TrackerKCF_create()
video = cv2.VideoCapture('kobe.mp4')

ok, frame = video.read()
bbox = cv2.selectROI(frame)
ok = tracker.init(frame, bbox)

while True:
    ok, frame = video.read()
    if not ok:
        break
    
    ok, bbox = tracker.update(frame)

    # bounding box printing
    if ok:
        (x, y, w, h) = [int(v) for v in bbox]
        cv2.rectangle(frame, (x,y), (x+w, y+h), (128,0,128), 2, cv2.LINE_8)
    else:
        cv2.putText(frame, "Error", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.imshow("KCF Tracking", frame)

    if cv2.waitKey(1) & 0XFF == 27:
        break