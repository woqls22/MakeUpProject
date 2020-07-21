import dlib, sys
import cv2
import numpy as np
from PIL import Image
import random
MAKE_TRANSPARENT = True

def draw_line(img, L):
  for i in range(len(L)-1):
    x = list(L[i])[0]+3
    y = list(L[i])[1]+3
    pallete=(255,255,255)
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
  #return img[fromY:maxY, minX:maxX]
  return img[:, :]

def is_face(item):
  # Decide whether the face Area is
  common_mask = 20
  #Gray Case
  if(((abs(item[0]-item[1])<common_mask) and (abs(item[1]-item[2])<common_mask) and (abs(item[2]-item[0])<common_mask))):
    return False
  #RGB Compare
  #236 200 188
  #241 179 172
  #185 121 114
  if(item[0]<item[1] or item[0]<item[2]):
    return False
  if(item[0]>item[1]+120 or item[0]>item[2]+120):
    return False
  if(item[0]<80):
    return False
  return True
def detect_nose_hole(item):
  #178 121 117
  #134 76 73
  #141 86 81
  #157 102 98
  #139 81 77
  if(item[0]<180 and item[1]<130 and item[2]<130):
    return False
  return True

def postprocess_face_layer(imgname, nose_x, nose_y):
  if (MAKE_TRANSPARENT):
    img = Image.open(imgname)
    img = img.convert("RGBA")
    datas = img.getdata()
    opencv_img = cv2.imread(imgname)
    width = opencv_img.shape[1]
    height = opencv_img.shape[0]
    opencv_img = cv2.cvtColor(opencv_img, cv2.COLOR_BGR2RGB)

    newData = []
    red_mask = 60
    mask_interval = 150
    common_mask = 30
    x = -1
    y = 0
    minx = min(nose_x)
    maxx = max(nose_x)
    miny = min(nose_y)
    maxy = max(nose_y)
    for item in datas:
      if(x>minx and x<maxx and y>miny and y<maxy):
        if(detect_nose_hole(item)):
          newData.append(item)
        else:
          newData.append((255, 255, 255, 0))
      else:
        newData.append(item)
      x = x + 1
      if (x >= width):
        y = y + 1
        x = 0
    img.putdata(newData)
    img.save(imgname, "PNG")
    print(imgname)

def get_rid_of_face_background(imgname, minX,maxX,maxY):
  if(MAKE_TRANSPARENT):
    img = Image.open(imgname)
    img = img.convert("RGBA")
    datas = img.getdata()
    opencv_img = cv2.imread(imgname)
    width = opencv_img.shape[1]
    height = opencv_img.shape[0]
    opencv_img = cv2.cvtColor(opencv_img, cv2.COLOR_BGR2RGB)
    color = [140,230,200]

    newData = []
    red_mask = 60
    mask_interval = 150
    common_mask = 30
    x=-1
    y=0
    for item in datas:
      #print("{0}, {1}, {2}".format(item[0],item[1],item[2]))
      #print("{0}, {1}, {2}".format(color[0],color[1],color[2]))
      #print("========================")
      #print(item)
      x=x+1
      if(y>maxY):
        if(is_face(item) or (x>minX and x<maxX)):
          #print(x)
          if(is_face(item)):
            newData.append(item)
          else:
            newData.append((255, 255, 255, 0))
            pass
        else:
          newData.append((255, 255, 255, 0))
      elif(item[0]<140 or item[2]>200 or ((abs(item[0]-item[1])<common_mask) and (abs(item[1]-item[2])<common_mask) and (abs(item[2]-item[0])<common_mask))):# Gray Filtering
        newData.append((255, 255, 255, 0))
      elif ((item[0] <= color[0]+red_mask and item[0]>color[0]-red_mask) or (item[1] <= color[1]+mask_interval or item[1]>color[1]-mask_interval) or (item[2] <= color[2]+mask_interval and item[2]>color[2]-mask_interval)):
        if ((item[0] > item[1] + 85) and (item[0] > item[2] + 85)):
          newData.append((255, 255, 255, 0))
        elif(item[0]>140 and item[1]<140 and item[2]<140):
          newData.append((255, 255, 255, 0))
        elif(item[0]>225 and item[1]<150 and item[2]<150):
          newData.append((255, 255, 255, 0))
        else:
           newData.append(item)
      elif((item[0]>item[1]+50) or (item[0]>item[2]+50)):
        newData.append((255, 255, 255, 0))
      elif(item[0]<item[2] and item[0]<item[1]):
        newData.append((255, 255, 255, 0))
      elif(item[1]<140 or item[2]<140):
        newData.append((255, 255, 255, 0))
      else:
        newData.append((255, 255, 255, 0))
      if (x >= width):
        y = y + 1
        x=0
    img.putdata(newData)
    img.save(imgname, "PNG")
    print(imgname)
    #opencv_img = cv2.imread(imgname)
    #cv2.GaussianBlur(opencv_img, (3,3),0)
    #cv2.imwrite(imgname, opencv_img)

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

def erase_layer(img,version, left_eyes_brow, right_eyes_brow, left_eyes, right_eyes, mouse):
  # PreProcessing
  if (MAKE_TRANSPARENT):
    cvImg = img
    #points1 = np.array(left_eyes_brow, np.int32)
    points2 = np.array(right_eyes_brow, np.int32)
    points3 = np.array(left_eyes, np.int32)
    points4 = np.array(right_eyes, np.int32)
    points5 = np.array(mouse,np.int32)
    #cvImg = cv2.fillConvexPoly(cvImg, points1, (0, 0, 0))
    #cvImg = cv2.fillConvexPoly(cvImg, points2, (0, 0, 0))
    cvImg = cv2.fillConvexPoly(cvImg, points3, (0, 0, 0))
    cvImg = cv2.fillConvexPoly(cvImg, points4, (0, 0, 0))
    #cvImg = cv2.fillConvexPoly(cvImg, points5, (0,0,0,))
    cv2.imwrite('./FacePart/face_line_img'+version+'.png', cvImg)
    #cv2.imshow('test', cvImg)



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
      elif(abs(item[0]-item[1])<20 and abs(item[1]-item[2])<20 and abs(item[2]-item[0])<20):
        newData.append((255, 255, 255, 0))
      else:
        newData.append(item)

    img.putdata(newData)
    img.save(imgname, "PNG")
    opencv_img = cv2.imread(imgname)
    cv2.GaussianBlur(opencv_img, (3,3),0)
    cv2.imwrite(imgname, opencv_img)

def get_cheek_layer(cheek_ori, left_cheekpoint, right_cheekpoint, face_line, nose):
  Lx, Ly = GetIntersetLeftPoints2D(left_cheekpoint)
  Rx, Ry = GetIntersetRightPoints2D(right_cheekpoint)
  Left_facedot = []
  Right_facedot = []
  cv2.line(cheek_ori, (Lx,Ly), (Lx,Ly), (0,0,0),5)
  cv2.line(cheek_ori, (Rx,Ry), (Rx,Ry), (0,0,0),5)
  i = 0
  Left_facedot.append([Lx,Ly])
  Right_facedot.append([Rx, Ry])
  oval_1=[]
  oval_2=[]
  for elem in face_line:
    if(i>=2 and i<=5 and i is not 3):
      Left_facedot.append(get_center_point(face_line[i-1],face_line[i]))
      Left_facedot.append(elem)
    if(i==3):
      Left_facedot.append(get_center_point(face_line[i-1],face_line[i]))
      Left_facedot.append(elem)
      oval_1.append(elem)
    if(i>=12 and i<=15 and i is not 3):
      Right_facedot.append(get_center_point(face_line[i-1],face_line[i]))
      Right_facedot.append(elem)
    if(i==13):
      Right_facedot.append(get_center_point(face_line[i-1],face_line[i]))
      Right_facedot.append(elem)
      oval_2.append(elem)
    i= i+1
  points1 = np.array(Left_facedot, np.int32)
  points2 = np.array(Right_facedot, np.int32)
  width = int((left_cheekpoint[1][0] - left_cheekpoint[0][0]) * 0.9)
  Lwidth = int((left_cheekpoint[1][0]-left_cheekpoint[0][0])*0.76)
  Rwidth = int((left_cheekpoint[1][0] - left_cheekpoint[0][0]) * 0.85)
  height = int((left_cheekpoint[1][1]-left_cheekpoint[0][1])*0.5)
  for i in range(0,len(points1)):
    cv2.ellipse(cheek_ori, tuple(get_center_point(points1[i], [Lx,Ly])), (Lwidth, height), -15, 0, 360, (0, 0, 255), -1)
  for i in range(0, len(points2)):
    cv2.ellipse(cheek_ori, tuple(get_center_point(points2[i], [Rx, Ry])), (Rwidth, height), 15, 0, 360, (0, 0, 255), -1)
  cheek_ori = cv2.fillConvexPoly(cheek_ori, points1,(0,0,255))
  cheek_ori = cv2.fillConvexPoly(cheek_ori, points2, (0,0,255))
  width = int(width*1.2)
  height= int(height*1.5)
  Lx = int(Lx-width*0.3)
  Rx = int(Rx+width*0.3)
  cv2.ellipse(cheek_ori, (Lx, Ly), (width, height), -20, 0, 360, (0, 0, 255), -1)
  cv2.ellipse(cheek_ori, (Rx, Ry), (width, height), 20, 0, 360, (0, 0, 255), -1)
  imgn = "CheekLayer.png"
  cv2.imwrite(imgn, cheek_ori)
  if (MAKE_TRANSPARENT):
    img = Image.open(imgn)
    img = img.convert("RGBA")
    datas = img.getdata()
    newData = []
    for item in datas:
      if (item[0]==255 and item[1]== 0 and item[2] == 0):
        newData.append(item)
      else:
        newData.append((255, 255, 255, 0))

    img.putdata(newData)
    imgname = "./FacePart/CheekLayer.png"
    print("Cheek Layer Extract [Path] : " + imgname)
    img.save(imgname, "PNG")

def get_center_point(P1,P2):
  x1 = int((P1[0]+P2[0])/2)
  y1 = int((P1[1]+P2[1])/2)
  return [x1,y1]
def GetIntersetLeftPoints2D(L):
  x1 = L[0][0]
  y1 = L[0][1]
  x2 = L[3][0]
  y2 = L[3][1]
  x3 = L[1][0]
  y3 = L[1][1]
  x4 = L[2][0]
  y4 = L[2][1]
  det = (((x1-x2)*(y3-y4))-((y1-y2)*(x3-x4)))
  Px = ((((x1*y2)-(y1*x2))*(x3-x4))-((x1-x2)*((x3*y4)-(y3*x4))))/det
  Py = ((((x1*y2)-(y1*x2))*(y3-y4))-((y1-y2)*((x3*y4)-(y3*x4))))/det
  return int(Px),int(Py)

def GetIntersetRightPoints2D(L):
  x1 = L[2][0]
  y1 = L[2][1]
  x2 = L[0][0]
  y2 = L[0][1]
  x3 = L[3][0]
  y3 = L[3][1]
  x4 = L[1][0]
  y4 = L[1][1]
  det = (((x1-x2)*(y3-y4))-((y1-y2)*(x3-x4)))
  Px = ((((x1*y2)-(y1*x2))*(x3-x4))-((x1-x2)*((x3*y4)-(y3*x4))))/det
  Py = ((((x1*y2)-(y1*x2))*(y3-y4))-((y1-y2)*((x3*y4)-(y3*x4))))/det
  return int(Px),int(Py)

def get_curve(Set):
  Set = list(Set)
  for i in range(1,len(Set)):
    Set.append(get_center_point(Set[i-1], Set[i]))
  return np.array(Set, np.int32)
