import cv2
import numpy as np
from cvzone.FaceMeshModule import FaceMeshDetector

# راه‌اندازی دوربین
cap = cv2.VideoCapture(0)
detector = FaceMeshDetector(maxFaces=2)  # شناسایی حداکثر 2 چهره

# تعریف رنگ‌های مختلف برای مثلث‌بندی چهره‌ها
colors = [(0, 255, 0), (255, 0, 0)]  # رنگ سبز برای چهره اول و آبی برای چهره دوم

while True:
    # خواندن فریم از دوربین
    success, img = cap.read()

    # پیدا کردن نقاط کلیدی چهره‌ها
    img, faces = detector.findFaceMesh(img, draw=False)

    if faces:
        # حلقه روی چهره‌ها (حداکثر 2 چهره)
        for i, face in enumerate(faces):
            # دریافت لیست نقاط کلیدی چهره
            points = np.array(face, np.int32)

            # ایجاد یک مستطیل محدودکننده برای مثلث‌بندی
            rect = cv2.boundingRect(points)

            # اعمال الگوریتم Delaunay Triangulation
            subdiv = cv2.Subdiv2D(rect)
            subdiv.insert(points.tolist())

            # دریافت مثلث‌های تولید شده
            triangles = subdiv.getTriangleList()
            triangles = np.array(triangles, dtype=np.int32)

            # انتخاب رنگ بر اساس شماره چهره
            color = colors[i % len(colors)]  # انتخاب رنگ چهره (اولی سبز، دومی آبی)

            # ترسیم مثلث‌ها روی تصویر
            for t in triangles:
                pt1 = (t[0], t[1])
                pt2 = (t[2], t[3])
                pt3 = (t[4], t[5])

                # رسم مثلث با رنگ مشخص
                cv2.line(img, pt1, pt2, color, 1)
                cv2.line(img, pt2, pt3, color, 1)
                cv2.line(img, pt3, pt1, color, 1)

    # نمایش تصویر
    cv2.imshow("Face Triangulation", img)

    # توقف با کلید 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
