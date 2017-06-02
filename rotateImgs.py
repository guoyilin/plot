#rotate img. 
import cv2 
import numpy as np 
import argparse 
 
 
parser = argparse.ArgumentParser(description='rotate img') 
parser.add_argument('imgsList', help='img list') 
parser.add_argument('savePath', help='save rotate imgs') 
args = parser.parse_args() 
 
angs = [-15] 
for line in open(args.imgsList, "r"): 
    img = cv2.imread(line.strip()) 
    imgName = line.strip().split("/")[-1].strip() 
    rows,cols = img.shape[:2] 
    for ang in angs: 
        M = cv2.getRotationMatrix2D((cols/2,rows/2),ang,1) 
        dst = cv2.warpAffine(img,M,(cols,rows)) 
        cv2.imwrite(args.savePath + "/" + imgName + "_rotate" + str(ang) + ".jpg", dst)
