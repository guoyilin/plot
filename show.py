#该笔记展示常见的显示图片的方法。

#cv2的方法： 不支持中文
img = cv.imread(imgs[imgName])
cv.rectangle(img, (x1,y1),(x2,y2),(0,0,255)) 
font = cv.FONT_HERSHEY_SIMPLEX
cv.putText(img,label,(x1,y1), font, 0.3,(255,255,255),1) 
cv.imwrite("train_data/" + imgName, img)


#PIL方法,支持中文
import PIL 
from PIL import Image, ImageDraw, ImageFont
font_path = os.environ.get('FONT_PATH', '/Library/Fonts/华文细黑.ttf')  #mac
font_path=os.environ.get("FONT_PATH", "/usr/share/fonts/arphicfonts/gbsn00lp.ttf") #linux
font_path= os.environ.get("FONT_PATH", "/usr/share/fonts/dejavu/DejaVuSansMono.ttf")#centos
ttFont = ImageFont.truetype(font_path, 25)

image = Image.open(orig_img)
draw = None 
try: 
       draw = ImageDraw.Draw(image) 
except IOError: 
       continue 
draw.rectangle((sheshi_x1, sheshi_y1, sheshi_x2, sheshi_y2), outline=(255,0,0)) 
draw.text((sheshi_x1 - 10, sheshi_y1 - 35), unicode(str(sheshi_tsrcode), 'utf-8'), (255,255,0), font = ttFont) 
image.save("")




