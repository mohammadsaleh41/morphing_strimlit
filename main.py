#کتابخانه اوپن سی وی
import cv2
# کتابخانه نامپای برای بخشی از کد که در مورد مختصات بعضی از نقاط صحبت می‌شه.
import numpy as np

# برای ساختن gif
from PIL import Image

#فراخوانی تابع FaceMeshDetector برای جلو گیری از هارد کد تک تک بخش‌های تصویر 
from cvzone.FaceMeshModule import FaceMeshDetector

# ساختن آبجکت meshdetector که در آینده نقاط حساس رو برای ما شناسایی می‌کنه.
meshdetector = FaceMeshDetector(maxFaces=1)

#فراخوانی تصاویر
img_1 = cv2.imread("Im371.jpg")
img_2 = cv2.imread("Im372.jpg") 

#تبدیل کانال رنگی تصاویر که در بین کار با opencv تصاویر با رنگ درست نمایش داده شود. و کانال آبی و قرمز جابه‌جا نمایش داده نشود.
image_1_p = cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
image_2_p = cv2.cvtColor(img_2, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

# گرفتن یک کپی اولیه برای حفظ تصاویر تغییر کانال داده شده اولیه.
image_1 = image_1_p.copy()
image_2 = image_2_p.copy()

#فراخوانی نقاط خاص تصویر یک و دو

frame_1, faces_1 = meshdetector.findFaceMesh(image_1)
frame_2, faces_2 = meshdetector.findFaceMesh(image_2)
# متغیرهای faces_1 و faces_2 شامل 468 نقطه خاص هستند که این نقاط در تمام تصاویر یکسان شناسایی می‌شوند. یعنی مثلا نقطه اول روی نوک بینی چهره داخل تصویر شناسایی می‌شود.

# اضافه کردن گوشه تصاویر به مجموعه نقاط مش چهره برای طراحی مثلث بندی‌ها
faces_1[0].append([0,0])

faces_1[0].append([0 , image_1.shape[0]] )

faces_1[0].append([image_1.shape[1] , 0] )

faces_1[0].append([image_1.shape[1] , image_1.shape[0] ])

faces_2[0].append([0,0])

faces_2[0].append([0 , image_2.shape[0]] )

faces_2[0].append([image_2.shape[1] , 0] )

faces_2[0].append([image_2.shape[1] , image_2.shape[0] ])


# مثلث بندی فعلی نقاط خاص این مثلث بندی‌ها می‌تواند بیشتر شود تا دقت و زیبایی کار بیشتر شود.
triangles_i = [
    [471 ,470 ,447 ],
    [471 ,469 ,152 ],
    # [447 ,152 ,10  ],
    [447 ,1   ,10  ],
    [447 ,1   ,152 ],
    [447 ,152 ,471 ],
    [469 ,127 ,152 ],
    [470 ,468 ,10  ],
    [469 ,468 ,127 ],
    [468 ,10  ,127 ],
    # [152 ,10  ,127 ],
    [152 ,1   ,127 ],
    [1   ,10  ,127 ],
    [447 ,10  ,470 ],
]
# در صورت بیشتر شدن مثلث بندی‌ها می‌توان پروژه‌هایی تقریبا شبیه به دیپ فیک را اجرا کرد.


# تبدیل ایندکس مثلث‌ها به مختصات گوشه‌های مثلث‌ها روی تصویر اولی
triangles_1 = []
for i in range(len(triangles_i)):
    triangle = np.array([faces_1[0][triangles_i[i][0]],
                faces_1[0][triangles_i[i][1]],
                faces_1[0][triangles_i[i][2]]])
    triangles_1.append(triangle)

# تبدیل ایندکس مثلث‌ها به مختصات گوشه‌های مثلث‌ها روی تصویر دومی
triangles_2 = []
for i in range(len(triangles_i)):
    triangle = np.array([faces_2[0][triangles_i[i][0]],
                faces_2[0][triangles_i[i][1]],
                faces_2[0][triangles_i[i][2]]])
    triangles_2.append(triangle)

# جدا کردن هر مثلث از داخل تصویر اول
triangles_image_1 = []
for i in range(len(triangles_i)):
    # create a mask
    
    triangle_cnt = triangles_1[i]
    mask = np.zeros(image_1.shape[:2], np.uint8)
    
    cv2.drawContours(mask, [triangle_cnt], 0, (255,255,255), -1)

    # compute the bitwise AND using the mask
    masked_img = cv2.bitwise_and(image_1_p,image_1_p,mask = mask)
    triangles_image_1.append(masked_img)

# جدا کردن هر مثلث از داخل تصویر دوم
triangles_image_2 = []
for i in range(len(triangles_i)):
    # create a mask
    
    triangle_cnt = triangles_2[i]
    mask = np.zeros(img_2.shape[:2], np.uint8)
    
    cv2.drawContours(mask, [triangle_cnt], 0, (255,255,255), -1)

    # compute the bitwise AND using the mask
    masked_img = cv2.bitwise_and(image_2_p,image_2_p,mask = mask)
    triangles_image_2.append(masked_img)

# لیست تصاویر n_t مرحله که از تصویر اول به تصویر دوم کم کم تبدیل می‌شوند
l_image_1_to_2 = []
n_t = 20
range_t = range(1 , n_t+1)
for t_i in range_t:
    t = t_i / n_t
    triangles_1_image_1_w = []
    for i in range(len(triangles_image_1)):
        srcPoints = np.float32(triangles_1[i])
        dstPoints = np.float32(triangles_2[i] * t + triangles_1[i]*(1-t))
        warp_mat = cv2.getAffineTransform(srcPoints, dstPoints)
        warp_dst = cv2.warpAffine(triangles_image_1[i], warp_mat, (img_1.shape[1], img_1.shape[0]))
        triangles_1_image_1_w.append(warp_dst)
    image_1_to_2 = triangles_1_image_1_w[0]
    for i in range(1 , len(triangles_1_image_1_w)):
        image_1_to_2 = cv2.addWeighted(image_1_to_2, 1, triangles_1_image_1_w[i], 1, 0.0)
    l_image_1_to_2.append(image_1_to_2)

# لیست تصاویر دوم به اول که در n_t مرحله تبدیل می‌شوند.
# فریم اول این لیست به شکلی هست که نقاط خاص چهره تصویر دوم در جایگاه نقاط خاص چهره تصویر اول هست.
# در فریم آخر این بخش نقاط خاص تصویر دوم به مکان اصلی خودش می‌رسه.
l_image_2_to_1 = []
for t_i in range_t:
    t = t_i / n_t
    triangles_2_image_2_w = []
    for i in range(len(triangles_image_2)):
        srcPoints = np.float32(triangles_2[i])
        dstPoints = np.float32(triangles_2[i] * t + triangles_1[i]*(1-t))
        warp_mat = cv2.getAffineTransform(srcPoints, dstPoints)
        warp_dst = cv2.warpAffine(triangles_image_2[i], warp_mat, (img_2.shape[1], img_2.shape[0]))
        triangles_2_image_2_w.append(warp_dst)
    image_2_to_1 = triangles_2_image_2_w[0]
    for i in range(1 , len(triangles_2_image_2_w)):
        image_2_to_1 = cv2.addWeighted(image_2_to_1, 1, triangles_2_image_2_w[i], 1, 0.0)
    l_image_2_to_1.append(image_2_to_1)

# ساختن لیستی n_t تایی از تصاویر جمع وزن دار تصویری از لیست اول و دوم هست. هرچقدر به آخرین مرحله نزدیک می‌شویم از وزن تصاویر لیست اول کم می‌شود و به وزن تصاویر لیست دوم اضافه می‌شود.
images = []
alpha = 1

for t_i in range_t:
    t = t_i / n_t
    
    img = cv2.addWeighted(l_image_2_to_1[t_i-1], t, l_image_1_to_2[t_i-1], 1-t, 0.0)
    images.append(img)

# ساختن gif از تصاویر موجود در لیست قبلی
def create_gif(images, gif_name):
    pil_images = [Image.fromarray(img) for img in images]  # Convert OpenCV images to PIL
    pil_images[0].save(
        gif_name, save_all=True, append_images=pil_images[1:], duration=500, loop=0
    )  # Save as GIF

# Path to save the GIF

gif_name = 'output_images.gif'
create_gif(images, gif_name)
