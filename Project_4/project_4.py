import os
import sys
import cv2
import matplotlib.pyplot
import numpy as np
from PIL import Image
from math import acos

def check_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path
def HSI_to_bgr(h, s, i):
    
    if 0 <= h <= 120 :
        
        b = i * (1 - s)
        r = i * (1 + (s * np.cos(np.radians(h)) / np.cos(np.radians(60) - np.radians(h))))
        g = i * 3 - (r + b)
    elif 120 < h <= 240:
        h -= 120
        r = i * (1 - s)
        g = i * (1 + (s * np.cos(np.radians(h)) / np.cos(np.radians(60) - np.radians(h))))
        b = 3 * i - (r + g)
    elif 0 < h <= 360:
        h -= 240
        g = i * (1 - s)
        b = i * (1 + (s * np.cos(np.radians(h)) / np.cos(np.radians(60) - np.radians(h))))
        r = i * 3 - (g + b)
    
    return [b, g, r]


def rgb_to_hue(b, g, r):
    if (b == g == r):
        return 0

    angle = 0.5 * ((r - g) + (r - b)) / np.sqrt(((r - g) ** 2) + (r - b) * (g - b))
    #print(angle)
    if b <= g:
        return np.ceil(np.degrees(np.arccos(angle)))
    else:
        return np.ceil(np.degrees(2 * np.pi - np.arccos(angle)))


def rgb_to_intensity(b, g, r):
    val = (b + g + r) / 3.
    if val == 0:
        return 0
    else:
        return val


def rgb_to_saturity(b, g, r):
    if r + g + b != 0:
        return 1. - 3. * np.min([r, g, b]) / (r + g + b)
    else:
        return 0

class myImage():
    """
    img -> cv2 read image -> BGR2RGB -> RGB2HSI 
    """
    def __init__(self, img_path, save):
        self.img = cv2.imread(img_path)
        self.rgb = self.bgr2rgb()
        self.hsi = self.rgb2hsi(self.img)
        self.save = save

    def bgr2rgb(self):
        rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        self.r, self.g, self.b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        return rgb
    def hsi2rgb(self, hsi_img):
        m, n = hsi_img.shape[:2]
        rgb = np.zeros((m, n, 3), dtype=np.uint8)
        I = np.zeros((m, n))
        S = np.zeros((m, n))
        H = np.zeros((m, n))
        for j in range(m):
            for k in range(n):
                h = hsi_img[j][k][0]
                s = hsi_img[j][k][1]
                i = hsi_img[j][k][2]
                
                bgr_tuple = HSI_to_bgr(h, s, i)
               # print(b, g, r)
                rgb[j][k][0] = np.clip(round(bgr_tuple[1] * 255.), 0, 255)
                rgb[j][k][1] = np.clip(round(bgr_tuple[2] * 255.), 0, 255)
                rgb[j][k][2] = np.clip(round(bgr_tuple[0] * 255.), 0, 255)
        return rgb
    
    def rgb2hsi(self, image):
        
        m, n = image.shape[:2]
        hsi = np.zeros((m, n, 3), dtype=np.float32)
        I = np.zeros((m, n))
        S = np.zeros((m, n))
        H = np.zeros((m, n))
        for j in range(m):
            for k in range(n):
                
                b = image[j][k][0] / 255.
                g = image[j][k][1] / 255.
                r = image[j][k][2] / 255.
                H[j][k] = rgb_to_hue(b, g, r)
                S[j][k] = rgb_to_saturity(b, g, r)
                I[j][k] = rgb_to_intensity(b, g, r)
                #print(H[j][k], S[j][k] ,I[j][k])
                #bgr_tuple = HSI_to_bgr(H[j][k], S[j][k], I[j][k])

                # hsi[j][k][0] = np.clip(round(bgr_tuple[0] * 255.), 0, 255)
                # hsi[j][k][1] = np.clip(round(bgr_tuple[1] * 255.), 0, 255)
                # hsi[j][k][2] = np.clip(round(bgr_tuple[2] * 255.), 0, 255)
                hsi[j][k][0] = H[j][k]
                hsi[j][k][1] = S[j][k]
                hsi[j][k][2] = I[j][k]
                #print(hsi)
        #print(hsi.shape)
        return hsi
    
    def sharpen_img(self, type="rgb"):
        sharpen_kernel = np.array([
            [-1, -1, -1],
            [-1,  9, -1],
            [-1, -1, -1]], dtype='float32')
        if type == 'rgb':
            img_sharpen = cv2.filter2D(self.rgb, -1, kernel=sharpen_kernel)
            return img_sharpen
        elif type == 'hsi':
            img_sharpen = self.hsi
            img_sharpen[..., -1] = cv2.filter2D(self.hsi[..., -1], -1, kernel=sharpen_kernel)
            return img_sharpen
        else:
            print("[Warn] Wrong sharpen type!!")


    def save_result(self):
        if self.save:
            img_list = [self.r, self.g, self.b, self.h, self.s, self.i]
            title_list = ["r", "g", "b", "h", "s", "i"]
            check_folder("result")
            for i in range(len(img_list)):
                image = Image.fromarray(img_list[i])
                if title_list[i] == "i":  # Binary form 
                    image = image.convert("L")
                image.save("result"+ '/' + title_list[i] + ".jpg", dpi=(200.0, 200.0, 0))

    def save_single_img(self, img, img_title):
        check_folder("result")
        image = Image.fromarray(np.uint8(img))
        image.save("result"+ '/' +img_title + ".jpg", dpi=(200.0, 200.0, 0))

def main():
    # Split the r, g, b and convert to h, s, i in class function
    img = myImage("LovePeace rose.tif", save=True)
    
    rgb_sharp = (img.sharpen_img("rgb"))
    hsi_sharp = img.sharpen_img("hsi")
    #print(hsi_sharp)
    test = img.hsi2rgb(hsi_sharp)
    #print(test)
    cv2.imshow("t", test)
    cv2.waitKey()
    # img.save_single_img(rgb_sharp, "rgb_sharp")
    # img.save_single_img(test, "hsi_sharpen")
    # img.save_result()
    print(img.hsi.shape)
    


if __name__ == "__main__":
    main()

