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
    image_1_p = cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB)
    image_2_p = cv2.cvtColor(img_2, cv2.COLOR_BGR2RGB)

    # استفاده از FaceMesh برای نقاط چهره
    meshdetector = FaceMeshDetector(maxFaces=1)
    frame_1, faces_1 = meshdetector.findFaceMesh(image_1_p)
    frame_2, faces_2 = meshdetector.findFaceMesh(image_2_p)

    # اضافه کردن گوشه‌های تصاویر
    faces_1[0].extend([[0, 0], [0, img_1.shape[0]], [img_1.shape[1], 0], [img_1.shape[1], img_1.shape[0]]])
    faces_2[0].extend([[0, 0], [0, img_2.shape[0]], [img_2.shape[1], 0], [img_2.shape[1], img_2.shape[0]]])

    # مثلث بندی
    triangles_i = [[471, 470, 447], [471, 469, 152], [447, 1, 10], [447, 1, 152], [447, 152, 471]]

    triangles_1, triangles_2 = [], []
    for i in range(len(triangles_i)):
        triangles_1.append(np.array([faces_1[0][j] for j in triangles_i[i]]))
        triangles_2.append(np.array([faces_2[0][j] for j in triangles_i[i]]))

    # تولید تصاویر میانی
    images = []
    n_t = 20
    for t_i in range(1, n_t+1):
        t = t_i / n_t
        img_inter = np.zeros_like(img_1)
        for tri_1, tri_2 in zip(triangles_1, triangles_2):
            warp_mat = cv2.getAffineTransform(np.float32(tri_1), np.float32(tri_2 * t + tri_1 * (1 - t)))
            warp_dst = cv2.warpAffine(img_1, warp_mat, (img_1.shape[1], img_1.shape[0]))
            img_inter = cv2.addWeighted(img_inter, 1, warp_dst, 1, 0)
        images.append(Image.fromarray(img_inter))

    # ذخیره به عنوان GIF
    gif_name = 'output.gif'
    images[0].save(gif_name, save_all=True, append_images=images[1:], duration=500, loop=0)

    st.image(gif_name, caption="GIF نهایی")
