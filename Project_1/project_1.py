
import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import openpyxl

def save_img(src, title, save_path):
    plt.figure()
    plt.imshow(src, cmap="gray")
    plt.title(title)
    plt.axis("off")
    
    if (os.path.exists(save_path + title + ".png")):
        return
    if is_save:
        plt.savefig(save_path + title + ".png", dpi=200, bbox_inches='tight')

def save_hist(src, title, save_path):
    plt.figure()
    plt.hist(src.ravel(), 256, [0, 255])
    plt.title(title.split()[1] + ' histogram')
    plt.xlabel("Brightness")
    
    if (os.path.exists(save_path + title.split()[1] + "-histogram.png")):
        return
    if is_save:
        plt.savefig(save_path + title.split()[1] + "-histogram.png", dpi=200, bbox_inches='tight')



def check_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def main(img, img_save_path):
    
    global is_save
    img_save_path = check_folder(img_save_path)
    image = cv2.imread(img, 0)
    

    
    if image is None:
        print("Could not open or find the image")
        sys.exit()


    kernel = np.array([[-1, -1, -1],
                       [-1,  8, -1],
                       [-1, -1, -1]], dtype="float32")

    g_x = np.array([[-1,  0,  1],
                    [-2 , 0,  2],
                    [-1 , 0,  1]], dtype="float32")

    g_y = np.array([[-1, -2, -1],
                    [0 ,  0,  0],
                    [1 ,  2,  1]], dtype="float32")


    smooth = (1 / 25) * np.ones((5, 5), dtype="float32")

    #=================================================================
    # Do the Laplacian with smoothing first
    smoothed_img = cv2.filter2D(image, -1, smooth)
    laplacian = abs(cv2.filter2D(smoothed_img, -1, kernel))
    

    #laplacian = cv2.filter2D(shift_image, -1, kernel)
    # if shift_image[:,:,:] >=255: shift_image = 255
    # print(shift_image)
    # Do the Laplacian without smoothing (uncomment the following line)
    # laplacian = cv2.filter2D(image, -1, kernel)
    #=================================================================

    laplacian_sharpen = image + laplacian

    gx_filtered = cv2.filter2D(image, -1, g_x)
    gy_filtered = cv2.filter2D(image, -1, g_y)

    sobel_gradient = abs(gx_filtered + gy_filtered)

    smoothed_gradient = cv2.filter2D(sobel_gradient, -1, smooth)

    f = smoothed_gradient * laplacian
    gamma = 0.5
    s = np.array(255*((f + image) / 255) ** gamma, dtype = 'uint8')
    
    
    img_list = [image, laplacian, laplacian_sharpen, sobel_gradient, smoothed_gradient, f, (f+image), s]
    img_title = ["(a) Original-image", "(b) Laplacian", "(c) Laplacian-sharpened", "(d) Sobel-gradient", "(e) smoothed-gradient", "(f) extracted feature", "(g)", "(h) final-image"]

    fig, ax = plt.subplots(2, 4, figsize=(16, 16))
    fig2, ax2 = plt.subplots(2, 4, figsize=(16, 16))
    for i in range(2):
        for j in range(4):
            ax[i, j].imshow(img_list[j + i*4], cmap="gray")
            ax[i, j].set_title(img_title[j + i*4])    
            ax[i, j].axis("off")
            ax2[i, j].hist(img_list[j + i*4].ravel(), 256, [0, 255])
            ax2[i, j].set_title(img_title[j + i*4]) 
    
    
    plt.show()
    save_hist(img_list[0], img_title[0], img_save_path)
    save_hist(img_list[-1], img_title[-1], img_save_path)



    origin_img_hist = np.histogram(image.ravel(), 256, [0,255])
    final_img_hist = np.histogram(s.ravel(), 256, [0,255])
    
    
    if not os.path.exists('Histograms-test2.xlsx'):
        df_ = pd.read_excel('Histograms-test2.xlsx', index_col=0)  
        
        if (img == image_kid):
            df_["p(r) \n(kid original)"] = origin_img_hist[0].tolist()
            df_["p(r) \n(kid output)"] = final_img_hist[0].tolist()
            writer = pd.ExcelWriter('Histograms-test2.xlsx', engine='openpyxl')
            # Convert the dataframe to an XlsxWriter Excel object.
            df_.to_excel(writer, sheet_name='Sheet1')
            writer.save()
        else:

            df_["p(r) \n(fruit original)"] = origin_img_hist[0].tolist()
            df_["p(r) \n(fruit output)"] = final_img_hist[0].tolist()
            writer = pd.ExcelWriter('Histograms-test2.xlsx', engine='openpyxl')
            # Convert the dataframe to an XlsxWriter Excel object.
            df_.to_excel(writer, sheet_name='Sheet1')
            writer.save()
            
    # save the images
    for i in range(len(img_list)):
        save_img(img_list[i], img_title[i], img_save_path)
    plt.close('all')
   


if __name__ == "__main__":

    image_fruit = "fruit_blurred_noisy.tif"
    image_kid = "kid_blurred-noisy.tif"
    is_save = True
    main(image_kid, "image_kid/")
    main(image_fruit, "image_fruit/")
    