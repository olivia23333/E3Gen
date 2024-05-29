import os
import imageio
from PIL import Image

img_all = []
for i in range(36):
    path = f'work_dirs/trans_viz/{str(i)}.jpg'
    img = Image.open(path)
    img_all.append(img)

img_save = img_all[17:] + img_all[:17]
# img_save = img_all
imageio.mimsave('work_dirs/trans_viz/tryon.mp4', img_save, codec='libx264')