import cv2, dlib, sys
import numpy as np
from PIL import Image
MAKE_TRANSPARENT = True
def draw_line(img, L):
  for i in range(len(L)-1):
    x = list(L[i])[0]+3
    y = list(L[i])[1]+3
    b,g,r = img[y,x]
    pallete = (int(b),int(g),int(r))
    #pallete=(255,255,255)
    try:
      cv2.line(img,(list(L[i])[0], list(L[i])[1]),(list(L[i+1])[0], list(L[i+1])[1]), color=pallete,thickness=4, lineType=cv2.LINE_AA)
    except:
      pass
# overlay function
def extract_part(img,part):
  x = []
  y = []
  for i in range(len(part)):
    x.append(part[0:len(part)][i][0])
    y.append(part[0:len(part)][i][1])
  minX = min(x)
  maxX = max(x)
  minY = min(y)
  maxY = max(y)
  return img[minY:maxY, minX:maxX]
def extract_nose_part(img,part):
  x = []
  y = []
  for i in range(len(part)):
    x.append(part[0:len(part)][i][0])
    y.append(part[0:len(part)][i][1])
  minX = min(x)
  maxX = max(x)
  minY = min(y)
  maxY = max(y)
  interval = int((maxX-minX)/3)
  return img[minY:maxY, minX-interval:maxX+interval]
def extract_eye_part(img,part):
  x = []
  y = []
  for i in range(len(part)):
    x.append(part[0:len(part)][i][0])
    y.append(part[0:len(part)][i][1])
  minX = min(x)
  maxX = max(x)
  minY = min(y)
  maxY = max(y)
  intervalY = int((maxY-minY)/2)
  intervalX = int((maxX-minX)/4)
  return img[minY-intervalY:maxY, minX-int(intervalX/2):maxX+intervalX]
def extract_face_part(img,part):
  x = []
  y = []
  for i in range(len(part)):
    x.append(part[0:len(part)][i][0])
    y.append(part[0:len(part)][i][1])
  minX = min(x)
  maxX = max(x)
  minY = min(y)
  maxY = max(y)
  fromY = 0
  if(int(minY-(maxY-minY)/1.8)>0):
    fromY = int(minY-(maxY-minY)/1.8)
  return img[fromY:int(maxY+((img.shape[0]-maxY)/2)), minX:maxX]
def get_rid_of_face_background(imgname):
  # 회색 계열 : R 146 G 141 B 141
  # 회색 계열 : R 218 G 216 B 216
  # 회색 계열 : R 203 G 200 B 200
  # 회색 계열 : R 233 G 232 B 231
  # 살색 계열 : R 229 G 190 B 182
  # 살색 계열 : R 207 G 165 B 159

  if(MAKE_TRANSPARENT):
    img = Image.open(imgname)
    img = img.convert("RGBA")
    datas = img.getdata()
    opencv_img = cv2.imread(imgname)
    width = opencv_img.shape[1]
    height = opencv_img.shape[0]
    opencv_img = cv2.cvtColor(opencv_img, cv2.COLOR_BGR2RGB)
    color = [230,190,180]

    newData = []
    red_mask = 150
    mask_interval = 150
    common_mask = 20

    for item in datas:
      #print("{0}, {1}, {2}".format(item[0],item[1],item[2]))
      #print("{0}, {1}, {2}".format(color[0],color[1],color[2]))
      #print("========================")
      if(item[0]<177 or item[2]>230):# Gray Filtering
        newData.append((255, 255, 255, 0))
      elif(item[0]==item[1] and item[1]==item[2]):
        newData.append((255, 255, 255, 0))
      elif ((item[0] <= color[0]+red_mask and item[0]>color[0]-red_mask) and (item[1] <= color[1]+mask_interval and item[1]>color[1]-mask_interval) and (item[2] <= color[2]+mask_interval and item[2]>color[2]-mask_interval)):
        newData.append(item)
        print(item)
      elif((abs(item[0]-item[1])<common_mask) and (abs(item[1]-item[2])<common_mask) and (abs(item[2]-item[0])<common_mask)):
        newData.append((255, 255, 255, 0))
      elif(item[0]<item[2] and item[0]<item[1]):
        newData.append((255, 255, 255, 0))
      elif(item[1]<140 or item[2]<140):
        newData.append((255, 255, 255, 0))
      else:
        newData.append((255, 255, 255, 0))
    img.putdata(newData)
    img.save(imgname, "PNG")
    opencv_img = cv2.imread(imgname)
    cv2.GaussianBlur(opencv_img, (3,3),0)
    cv2.imwrite(imgname, opencv_img)
    print("Transparent Task Done.")

    '''
     print(mouse)
     print(left_eyes)
     print(right_eyes)
     print(nose)
     print(face_line)
     print(left_eyes_brow)
     print(right_eyes_brow)
     '''
def get_Crop_point(part):
  x = []
  y = []
  for i in range(len(part)):
    x.append(part[0:len(part)][i][0])
    y.append(part[0:len(part)][i][1])
  minX = min(x)
  maxX = max(x)
  minY = min(y)
  maxY = max(y)
  return minX, maxX, minY, maxY
def is_in_Area(x,y,minX,maxX,minY,maxY):
  if(x<=maxX and x>=minX and y>=minY and y<=maxY):
    return True
  else:
    return False
def get_face_layer(imgname, mouse, left_eyes,right_eyes,nose,face_line,left_eyes_brow,right_eyes_brow):
  if(MAKE_TRANSPARENT):
    img = cv2.imread(imgname)
    mouse = mouse-mouse.min(axis=0)
    left_eyes = left_eyes - left_eyes.min(axis=0)
    right_eyes = right_eyes - right_eyes.min(axis=0)
    nose = nose - nose.min(axis=0)
    face_line = face_line - face_line.min(axis=0)
    left_eyes_brow = left_eyes_brow - left_eyes_brow.min(axis=0)
    right_eyes_brow = right_eyes_brow - right_eyes_brow.min(axis=0)

    rect_mouse = cv2.boundingRect(mouse)
    x, y, w, h = rect_mouse
    croped = img[y:y + h, x:x + w].copy()
    mask = np.zeros(croped.shape[:2], np.uint8)
    cv2.drawContours(mask, [mouse], -1, (255, 255, 255), -1, cv2.LINE_AA)
    dst = cv2.bitwise_and(croped, croped, mask=mask)

    rect_left_eyes = cv2.boundingRect(left_eyes)
    x, y, w, h = rect_left_eyes
    croped = img[y:y + h, x:x + w].copy()
    mask = np.zeros(croped.shape[:2], np.uint8)
    cv2.drawContours(mask, [left_eyes], -1, (255, 255, 255), -1, cv2.LINE_AA)
    dst = cv2.bitwise_and(croped, croped, mask=mask)
    rect_right_eyes = cv2.boundingRect(right_eyes)
    x, y, w, h = rect_right_eyes
    croped = img[y:y + h, x:x + w].copy()
    mask = np.zeros(croped.shape[:2], np.uint8)
    cv2.drawContours(mask, [right_eyes], -1, (255, 255, 255), -1, cv2.LINE_AA)
    dst = cv2.bitwise_and(croped, croped, mask=mask)
    rect_nose = cv2.boundingRect(nose)
    x, y, w, h = rect_nose
    croped = img[y:y + h, x:x + w].copy()
    mask = np.zeros(croped.shape[:2], np.uint8)
    cv2.drawContours(mask, [nose], -1, (255, 255, 255), -1, cv2.LINE_AA)
    dst = cv2.bitwise_and(croped, croped, mask=mask)
    rect_face_line = cv2.boundingRect(face_line)
    x, y, w, h = rect_face_line
    croped = img[y:y + h, x:x + w].copy()
    mask = np.zeros(croped.shape[:2], np.uint8)
    cv2.drawContours(mask, [face_line], -1, (255, 255, 255), -1, cv2.LINE_AA)
    dst = cv2.bitwise_and(croped, croped, mask=mask)
    rect_left_eyes_brow = cv2.boundingRect(left_eyes_brow)
    x, y, w, h = rect_left_eyes_brow
    croped = img[y:y + h, x:x + w].copy()
    mask = np.zeros(croped.shape[:2], np.uint8)
    cv2.drawContours(mask, [left_eyes_brow], -1, (255, 255, 255), -1, cv2.LINE_AA)
    dst = cv2.bitwise_and(croped, croped, mask=mask)
    rect_right_eyes_brow = cv2.boundingRect(right_eyes_brow)
    x, y, w, h = rect_right_eyes_brow
    croped = img[y:y + h, x:x + w].copy()
    mask = np.zeros(croped.shape[:2], np.uint8)
    cv2.drawContours(mask, [right_eyes_brow], -1, (255, 255, 255), -1, cv2.LINE_AA)
    dst = cv2.bitwise_and(croped, croped, mask=mask)

    bg = np.ones_like(croped, np.uint8)*255
    cv2.bitwise_not(bg,bg, mask=mask)
    dst2 = bg+ dst

def get_rid_of_background(imgname):
  # 회색 계열 : R 146 G 141 B 141
  # 회색 계열 : R 218 G 216 B 216
  # 회색 계열 : R 203 G 200 B 200
  # 회색 계열 : R 233 G 232 B 231
  # 살색 계열 : R 229 G 190 B 182
  # 살색 계열 : R 207 G 165 B 159
  if(MAKE_TRANSPARENT):
    img = Image.open(imgname)
    img = img.convert("RGBA")
    datas = img.getdata()
    opencv_img = cv2.imread(imgname)
    opencv_img = cv2.cvtColor(opencv_img, cv2.COLOR_BGR2RGB)
    color = [229,190,182]

    newData = []
    red_mask = 50
    mask_interval = 130
    for item in datas:
      #print("{0}, {1}, {2}".format(item[0],item[1],item[2]))
      #print("{0}, {1}, {2}".format(color[0],color[1],color[2]))
      #print("========================")
      if ((item[0] <= color[0]+red_mask and item[0]>color[0]-red_mask) and (item[1] <= color[1]+mask_interval and item[1]>color[1]-mask_interval) and (item[2] <= color[2]+mask_interval and item[2]>color[2]-mask_interval)):
        newData.append((255, 255, 255, 0))
      elif(abs(item[0]-item[1])<10 and abs(item[1]-item[2])<10 and abs(item[2]-item[0])<10):
        newData.append((255, 255, 255, 0))
      else:
        newData.append(item)

    img.putdata(newData)
    img.save(imgname, "PNG")
    opencv_img = cv2.imread(imgname)
    cv2.GaussianBlur(opencv_img, (3,3),0)
    cv2.imwrite(imgname, opencv_img)

    print("Transparent Task Done.")