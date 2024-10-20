import streamlit as st
import cv2
import numpy as np
from PIL import Image
from cvzone.FaceMeshModule import FaceMeshDetector

st.title("تبدیل دو تصویر به GIF")

# آپلود باکس‌ها برای دریافت دو تصویر
img_file_1 = st.file_uploader("تصویر اول را آپلود کنید", type=["jpg", "png"])
img_file_2 = st.file_uploader("تصویر دوم را آپلود کنید", type=["jpg", "png"])

if img_file_1 and img_file_2:
    # خواندن تصاویر از ورودی آپلود
    img_1 = np.array(Image.open(img_file_1))
    img_2 = np.array(Image.open(img_file_2))

    # تبدیل به RGB
    image_1_p = img_1.copy()
    image_2_p = img_2.copy()
    image_1 = image_1_p.copy()
    image_2 = image_2_p.copy()

    # استفاده از FaceMesh برای نقاط چهره
    meshdetector = FaceMeshDetector(maxFaces=1)
    frame_1, faces_1 = meshdetector.findFaceMesh(image_1)
    frame_2, faces_2 = meshdetector.findFaceMesh(image_2)

    # اضافه کردن گوشه‌های تصاویر
    faces_1[0].extend([[0, 0], [0, img_1.shape[0]], [img_1.shape[1], 0], [img_1.shape[1], img_1.shape[0]]])
    faces_2[0].extend([[0, 0], [0, img_2.shape[0]], [img_2.shape[1], 0], [img_2.shape[1], img_2.shape[0]]])

    # مثلث بندی
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
    triangles_1, triangles_2 = [], []
    for i in range(len(triangles_i)):
        triangles_1.append(np.array([faces_1[0][j] for j in triangles_i[i]]))
        triangles_2.append(np.array([faces_2[0][j] for j in triangles_i[i]]))

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
        images.append(Image.fromarray(img))

    # ذخیره به عنوان GIF
    gif_name = 'output.gif'
    images[0].save(gif_name, save_all=True, append_images=images[1:], duration=100, loop=0)

    st.image(gif_name, caption="GIF نهایی")
