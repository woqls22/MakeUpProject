import dlib
import cv2
import numpy as np
from PIL import Image
import math
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
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
  common_mask = 10
  #250 146 121
  #Gray Case
  if(((abs(item[0]-item[1])<common_mask) and (abs(item[1]-item[2])<common_mask) and (abs(item[2]-item[0])<common_mask))):
    return False
  if(item[0]<120):
    return False
  if(item[0]<item[1] and item[0]<item[2]):
    return False
  if(item[0]>item[1]+120):
    return False
  return True

def detect_nose_hole(item):
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

def get_rid_of_face_background(imgname, minX,maxX,maxY):
  maxY = maxY+int(maxY*0.2)

  img = cv2.imread(imgname)

  img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  # skin color range for hsv color space
  HSV_mask = cv2.inRange(img_HSV, (0, 15, 0), (17, 170, 255))
  HSV_mask = cv2.morphologyEx(HSV_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

  # converting from gbr to YCbCr color space
  img_YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
  # skin color range for hsv color space
  YCrCb_mask = cv2.inRange(img_YCrCb, (0, 80, 85), (255, 180, 135))
  YCrCb_mask = cv2.morphologyEx(YCrCb_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

  # merge skin detection (YCbCr and hsv)
  global_mask = cv2.bitwise_and(YCrCb_mask, HSV_mask)
  global_mask = cv2.medianBlur(global_mask, 3)
  global_mask = cv2.morphologyEx(global_mask, cv2.MORPH_OPEN, np.ones((4, 4), np.uint8))

  global_result = cv2.bitwise_and(img, img, mask=global_mask)
  cv2.imwrite("./output/SkinLayer.png", global_result)

  if(MAKE_TRANSPARENT):
    opencv_img = cv2.imread(imgname)
    width = opencv_img.shape[1]
    height = opencv_img.shape[0]
    opencv_img = cv2.GaussianBlur(opencv_img, (3, 3), 2)
    #opencv_img = cv2.cvtColor(opencv_img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(imgname, opencv_img)
    img = Image.open(imgname)
    img = img.convert("RGBA")
    datas = img.getdata()

    color = [140,230,200]

    newData = []
    red_mask = 60
    mask_interval = 150
    common_mask = 30
    x=-1
    y=0
    for item in datas:
      x=x+1
      if(is_face(item) or (x>minX and x<maxX)):
        if(is_face(item) and not (item[0]==255 and item[1] == 255 and item[2]==255)):# and not (y>maxY)
          R = item[0]
          G = item[1]
          B = item[2]
          A = item[3]
          newData.append((R+20,G+20,B+20,A))
        else:
          newData.append((255, 255, 255, 0))
          pass
      else:
        newData.append((255, 255, 255, 0))
      if (x >= width):
        y = y + 1
        x = 0
    img.putdata(newData)
    img.save(imgname, "PNG")
    # 가우시안 블러 처리 코드.
    # face = cv2.imread(imgname)
    # face = cv2.GaussianBlur(face, (2,2), 2)
    # cv2.imwrite(imgname,face)
    # img = Image.open(imgname)
    # img = img.convert("RGBA")
    # width = face.shape[1]
    # height = face.shape[0]
    # datas = img.getdata()
    # newData = []
    # x=-1
    # y=0
    # for item in datas:
    #   x=x+1
    #   if(item[0]==255 and item[1]==255 and item[2] ==255):
    #     newData.append((255,255,255,0))
    #   else:
    #     newData.append(item)
    #   if (x >= width):
    #     y = y + 1
    #     x = 0
    # img.putdata(newData)
    # img.save(imgname,"PNG")
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
    cv2.imwrite('./output/face_line_img'+version+'.png', cvImg)
    #cv2.imshow('test', cvImg)



def get_cheek_layer(cheek_ori, left_cheekpoint, right_cheekpoint, face_line, center_point, R,G,B):
  Lx, Ly = GetIntersetLeftPoints2D(left_cheekpoint)
  Rx, Ry = GetIntersetRightPoints2D(right_cheekpoint)
  inter_val = 0.4 # 중심 값 이동 변수'
  max_alpha = 40
  weight_X = 1.4 # X좌표 가중치. 값이 커질수록 X축기준 흐려짐 심함
  weight_Y = 1  # Y 좌표 가중치. 값이 커질수록 Y축기준 흐려짐 심함
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
    cv2.ellipse(cheek_ori, tuple(get_center_point(points1[i], [Lx,Ly])), (Lwidth, height), -15, 0, 360, (0,0,255), -1)
  for i in range(0, len(points2)):
    cv2.ellipse(cheek_ori, tuple(get_center_point(points2[i], [Rx, Ry])), (Rwidth, height), 15, 0, 360, (0,0,255), -1)
  cheek_ori = cv2.fillConvexPoly(cheek_ori, points1,(0,0,255))
  cheek_ori = cv2.fillConvexPoly(cheek_ori, points2, (0,0,255))
  width = int(width*1.2)
  height= int(height*1.5)
  Lx = int(Lx-width*0.3)
  Rx = int(Rx+width*0.3)
  alpha_left = [int(Lx-width*inter_val),Ly]
  alpha_right = [int(Rx+width*inter_val),Ry]
  cv2.ellipse(cheek_ori, (Lx, Ly), (width, height), -20, 0, 360, (0,0,255), -1)
  cv2.ellipse(cheek_ori, (Rx, Ry), (width, height), 20, 0, 360, (0,0,255), -1)
  imgn = "./output/CheekLayer.png"
  cv2.imwrite(imgn, cheek_ori)
  if (MAKE_TRANSPARENT):
    img = Image.open(imgn)
    opencv_img = cv2.imread(imgn)
    width = opencv_img.shape[1]
    height = opencv_img.shape[0]
    img = img.convert("RGBA")
    x=-1
    y=0
    datas = img.getdata()
    newData = []
    min_val ,max_val = cal_min_max(imgn,alpha_right[0],alpha_right[1],alpha_left[0],alpha_left[1],center_point,B,G,R) #정규화를 위한 최대 최소 계산
    for item in datas:
      x=x+1
      if (item[0]==255 and item[1]== 0and item[2] == 0 and x <center_point):
        distance = math.sqrt(((alpha_left[0]-x)*weight_X)**2+((alpha_left[1]-y)*weight_Y)**2)
        #print("Right distance : "+str(normalization(distance,max_val,min_val)*100)) #정규화 값 출력
        newData.append((R,G,B,int(max_alpha-normalization(distance,max_val,min_val)*100))) # 거리비례 정규화 역순
      elif(item[0]==255 and item[1]== 0 and item[2] == 0  and x >= center_point):
        distance = math.sqrt(((alpha_right[0]-x)*weight_X)**2+((alpha_right[1]-y)*weight_Y)**2)
        newData.append((R,G,B, int(max_alpha-normalization(distance,max_val,min_val)*100)))
      else:
        newData.append((255, 255, 255, 0))
      if (x >= width):
        y = y + 1
        x = 0
    img.putdata(newData)
    imgname = "./output/CheekLayer.png"

    print("Cheek Layer Extract [Path] : " + imgname)
    img.save(imgname, "PNG")
def normalization(distance,Max_val, Min_val): #정규화 max, min 사이 값으로 변경
  return (distance-Min_val)/(Max_val-Min_val)

def cal_min_max(fname,Rx,Ry,Lx,Ly,center_point, B,G,R):
  img = Image.open(fname)
  opencv_img = cv2.imread(fname)
  width = opencv_img.shape[1]
  img = img.convert("RGBA")
  x = -1
  y = 0
  datas = img.getdata()
  distance=[]
  for item in datas:
    x = x + 1
    if (item[0] == 255 and item[1] == 0 and item[2] == 0 and x < center_point):
      distance.append(math.sqrt((Lx-x)**2+(Ly-y)**2))
    elif (item[0] == 255 and item[1] ==0 and item[2] == 0 and x >= center_point):
      distance.append(math.sqrt((Rx-x)**2+(Ry-y)**2))
    if (x >= width):
      y = y + 1
      x = 0
  return min(distance),max(distance)


def eyebrow_cal_min_max(fname,Rx,Ry,Lx,Ly,center_point):
  img = Image.open(fname)
  opencv_img = cv2.imread(fname)
  width = opencv_img.shape[1]
  img = img.convert("RGBA")
  x = -1
  y = 0
  datas = img.getdata()
  distance=[]
  for item in datas:
    x = x + 1
    if (item[0] == 255 and item[1] == 255 and item[2] == 255 and x < center_point):
      distance.append(math.sqrt((Lx-x)**2+(Ly-y)**2))
    elif (item[0] == 255 and item[1] ==255 and item[2] == 255 and x >= center_point):
      distance.append(math.sqrt((Rx-x)**2+(Ry-y)**2))
    if (x >= width):
      y = y + 1
      x = 0
  return min(distance),max(distance)

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

def cal_min_max_center(fname,Cx,Cy):
  img = Image.open(fname)
  opencv_img = cv2.imread(fname)
  width = opencv_img.shape[1]
  height = opencv_img.shape[0]
  img = img.convert("RGBA")
  x = -1
  y = 0
  datas = img.getdata()
  distance=[]
  for item in datas:
    x = x + 1
    distance.append(int(math.sqrt((Cx-x)**2+(Cy-y)**2)))
    if (x >= width):
      y = y + 1
      x = 0
  return min(distance),max(distance)

def get_lip_layer(imgname, mouse,R,G,B):
  img = cv2.imread(imgname)
  img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  mouse_mask = np.zeros_like(img_gray)

  x = -1
  y = 0
  x_list=[]
  y_list=[]
  max_alpha = 100
  for i in mouse:
    x_list.append(i[0])
    y_list.append(i[1])
  min_x = min(x_list)-5
  max_x = max(x_list)+5
  min_y = min(y_list)-3
  max_y = max(y_list)+5

  mouse = np.array(mouse, np.int32)
  mouse_mask = cv2.fillPoly(mouse_mask, [mouse], (255, 255, 255))

  cv2.imwrite("mouse_mask.png",mouse_mask)

  if(MAKE_TRANSPARENT):
    img = Image.open("mouse_mask.png")
    img = img.convert("RGBA")
    datas = img.getdata()
    opencv_img = cv2.imread("mouse_mask.png")
    opencv_img = cv2.cvtColor(opencv_img, cv2.COLOR_BGR2RGB)
    width = opencv_img.shape[1]
    height = opencv_img.shape[0]
    newData = []
    x = -1
    y = 0

    center_x = int((min_x+max_x)/2)
    center_y = int((min_y + max_y) / 2)
    min_val, max_val = cal_min_max_center(imgname, center_x,center_y)  # 정규화를 위한 최대 최소 계산
    centerLip_color = opencv_img[center_y,center_x]
    for item in datas:
      x = x + 1
      #and Lip_Similar_with_point(centerLip_color, item)
      if (not (item[0]==255 and item[1] == 255 and item[2] == 255)):
        newData.append((255, 255, 255, 0))
      else:
        distance = math.sqrt(abs(((center_x - x)*0.45) ** 4) + abs(((center_y - y)*0.85) ** 4))
        newData.append((R, G, B, int(max_alpha - normalization(distance, max_val, min_val)*100)))
      if (x >= width):
        y = y + 1
        x = 0

    img.putdata(newData)
    imgname = "./output/lip_layer.png"
    print("lip  Layer Extract [Path] : ./output/lip_layer.png")
    img.save(imgname, "PNG")

def get_eyebrow_layer(imgname, lefteyebrow, righteyebrow):
  x = -1
  y = 0
  l_x_list=[]
  l_y_list=[]
  for i in lefteyebrow:
    l_x_list.append(i[0])
    l_y_list.append(i[1])
  lefteyebrowmin_x = min(l_x_list)-5
  lefteyebrowmax_x = max(l_x_list)+5
  lefteyebrowmin_y = min(l_y_list)-3
  lefteyebrowmax_y = max(l_y_list)+5

  r_x_list = []
  r_y_list = []
  for i in righteyebrow:
    r_x_list.append(i[0])
    r_y_list.append(i[1])
  righteyebrowmin_x = min(r_x_list) - 5
  righteyebrowmax_x = max(r_x_list) + 5
  righteyebrowmin_y = min(r_y_list) - 3
  righteyebrowmax_y = max(r_y_list) + 5
  if(MAKE_TRANSPARENT):
    img = Image.open(imgname)
    img = img.convert("RGBA")
    datas = img.getdata()
    opencv_img = cv2.imread(imgname)
    opencv_img = cv2.cvtColor(opencv_img, cv2.COLOR_BGR2RGB)
    width = opencv_img.shape[1]
    height = opencv_img.shape[0]
    newData = []
    x = -1
    y = 0
    l_center_x = int((lefteyebrowmin_x + lefteyebrowmax_x)/2)
    l_center_y = int((lefteyebrowmin_y + lefteyebrowmax_y) / 2)
    left_center_eyebrow_color = opencv_img[l_center_y,l_center_x]

    center_x = int((righteyebrowmin_x+righteyebrowmax_x)/2)
    center_y = int((righteyebrowmin_y + righteyebrowmax_y) / 2)
    right_center_eyebrow_color = opencv_img[center_y,center_x]


    for item in datas:
      if((is_in_Area(x,y,lefteyebrowmin_x,lefteyebrowmax_x,lefteyebrowmin_y,lefteyebrowmax_y) and Eyebrow_Similar_with_point(left_center_eyebrow_color, item))):
        newData.append((255,255,255,60))
      elif((is_in_Area(x, y, righteyebrowmin_x, righteyebrowmax_x, righteyebrowmin_y, righteyebrowmax_y) and (Eyebrow_Similar_with_point(right_center_eyebrow_color, item)))):
        newData.append((255,255,255,60))
      else:
        newData.append((255,255,255,0))
      x = x + 1
      if (x >= width):
        y = y + 1
        x = 0
    img.putdata(newData)
    imgname = "./output/eyebrow.png"
    img.save(imgname, "PNG")


def Lip_Similar_with_point(source_color, compare_obj):
   red_confidence = 50
   confidence = 110 # For Filtering Gray
   #print(source_color)
   sorce_r = source_color[0]
   sorce_g = source_color[1]
   sorce_b = source_color[2]
   obj_r = compare_obj[0]
   obj_g = compare_obj[1]
   obj_b = compare_obj[2]
   if(obj_b>170 or obj_g>170):
     return False
   if(abs(sorce_r - obj_r)<red_confidence and abs(sorce_g - obj_g)<confidence and abs(sorce_b - obj_b)<confidence):
     return True
   if(abs(sorce_r - obj_g)>60 and abs(sorce_r - obj_b)>60):
     return True

   return False
def Eyebrow_Similar_with_point(source_color, compare_obj):
  red_confidence = 40
  confidence = 130
  sorce_r = source_color[0]
  sorce_g = source_color[1]
  sorce_b = source_color[2]
  obj_r = compare_obj[0]
  obj_g = compare_obj[1]
  obj_b = compare_obj[2]
  if (abs(sorce_r - obj_r) < red_confidence and abs(sorce_g - obj_g) < confidence):
    return True
  if (abs(sorce_r - obj_r) < red_confidence and abs(sorce_b - obj_b) < confidence):
    return True
  if(obj_r<obj_g and obj_r< obj_b):
    return True
  return False

def eyemasking(origin_src, out_addr):
  img = cv2.imread(origin_src)
  img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  eye_mask = np.zeros_like(img_gray)

  detector = dlib.get_frontal_face_detector()
  predictor = dlib.shape_predictor("shape_predictor_194_face_landmarks.dat")
  faces = detector(img_gray)

  number = 1

  for face in faces:
    landmarks = predictor(img_gray, face)
    landmarks_points = []

    # 왼쪽 눈
    for n in range(48, 54):
      x = landmarks.part(n).x
      y = landmarks.part(n).y * number
      landmarks_points.append((x, y))
    for n in range(55, 65):
      x = landmarks.part(n).x
      y = landmarks.part(n).y * number
      landmarks_points.append((x, y))
    for n in range(66, 70):
      x = landmarks.part(n).x
      y = landmarks.part(n).y * number
      landmarks_points.append((x, y))

    # 오른쪽 눈
    for n in range(26, 32):
      x = landmarks.part(n).x
      y = landmarks.part(n).y * number
      landmarks_points.append((x, y))
    for n in range(33, 43):
      x = landmarks.part(n).x
      y = landmarks.part(n).y * number
      landmarks_points.append((x, y))
    for n in range(44, 48):
      x = landmarks.part(n).x
      y = landmarks.part(n).y * number
      landmarks_points.append((x, y))

      # cv2.circle(img, (x, y), 3, (0, 0, 255), -1)

    points = np.array(landmarks_points, np.int32)

    convexhull = []
    convexhull.append(cv2.convexHull(points[0:18]))
    convexhull.append(cv2.convexHull(points[20:38]))

    for con in convexhull:
      cv2.polylines(img, [con], True, (255, 0, 0), 1)
      cv2.fillConvexPoly(eye_mask, con, 255)

    eye_image_1 = cv2.bitwise_and(img, img, mask=eye_mask)

    # Delaunay triangluation
    rect = []
    for con in convexhull:
      rect.append(cv2.boundingRect(con))

    subdiv = []
    triangles = []

    for r in rect:
      subdiv.append(cv2.Subdiv2D(r))

    for i in range(0, len(subdiv)):
      if i == 0:
        subdiv[i].insert(landmarks_points[0:18])
      else:
        subdiv[i].insert(landmarks_points[20:38])

      triangles.append(subdiv[i].getTriangleList())
      triangles[i] = np.array(triangles[i], dtype=np.int32)

    # print(triangles)
    for i in range(0, len(triangles)):
      for t in triangles[i]:
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])

        cv2.line(img, pt1, pt2, (0, 0, 255), 2)
        cv2.line(img, pt2, pt3, (0, 0, 255), 2)
        cv2.line(img, pt1, pt3, (0, 0, 255), 2)
    cv2.imwrite(out_addr, eye_mask)
    img = Image.open(out_addr)  # 파일 열기
    img = img.convert("RGBA")  # RGBA형식으로 변환
    datas = img.getdata()  # datas에 일차원 배열 형식으로 RGBA입력
    newData = []
    for item in datas:
      if (item[0] == 255 and item[1] == 255 and item[2] == 255):  # 해당 픽셀 색이 흰색이면
        newData.append(item)  # 해당 영역 추가
      else:  # 그렇지 않으면
        newData.append((255, 255, 255, 0))  # 투명 추가
    img.putdata(newData)  # 데이터 입력
    img.save(out_addr)  # 이미지name으로 저장

def eyeshadow_masking(origin_src, out_addr):
  img = cv2.imread(origin_src)

  img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  eyeshadow_mask = np.zeros_like(img_gray)

  detector = dlib.get_frontal_face_detector()
  predictor = dlib.shape_predictor("shape_predictor_194_face_landmarks.dat")
  faces = detector(img_gray)

  for face in faces:
    landmarks = predictor(img_gray, face)
    landmarks_points = []

    ##########################   left   ################################

    # 0
    x = landmarks.part(106).x * 0.97
    y = landmarks.part(106).y * 1.05
    landmarks_points.append((x, y))
    # 1
    x = landmarks.part(107).x
    y = landmarks.part(107).y * 1.01
    landmarks_points.append((x, y))
    # 2
    x = landmarks.part(108).x
    y = landmarks.part(108).y * 1.005
    landmarks_points.append((x, y))
    # 3
    x = landmarks.part(110).x
    y = landmarks.part(110).y
    landmarks_points.append((x, y))
    # 4
    x = landmarks.part(111).x
    y = landmarks.part(111).y * 1.01
    landmarks_points.append((x, y))
    # 5
    x = landmarks.part(112).x
    y = landmarks.part(112).y * 1.02
    landmarks_points.append((x, y))

    # 6
    x = landmarks.part(48).x * 1.05
    y = landmarks.part(48).y * 1.01
    landmarks_points.append((x, y))

    # 7~12
    for n in range(48, 54):
      if n == 48:
        x = landmarks.part(n).x * 1.007
        y = landmarks.part(n).y * 0.995
      elif n == 49:
        x = landmarks.part(n).x
        y = landmarks.part(n).y * 0.995
      else:
        x = landmarks.part(n).x * 0.995
        y = landmarks.part(n).y * 0.995
      landmarks_points.append((x, y))
    # 13~17
    for n in range(55, 60):
      if n == 59:
        x = landmarks.part(n).x * 0.855
        y = landmarks.part(n).y * 1.01
      elif n == 58:
        x = landmarks.part(n).x * 0.97
        y = landmarks.part(n).y * 0.995
      else:
        x = landmarks.part(n).x * 0.993
        y = landmarks.part(n).y * 0.993
      landmarks_points.append((x, y))

    # 18
    x = landmarks.part(113).x
    y = landmarks.part(113).y * 1.05
    landmarks_points.append((x, y))

    ##########################   right   ################################

    # 19
    x = landmarks.part(84).x * 1.01
    y = landmarks.part(84).y * 1.05
    landmarks_points.append((x, y))
    # 20
    x = landmarks.part(85).x
    y = landmarks.part(85).y * 1.015
    landmarks_points.append((x, y))
    # 21
    x = landmarks.part(86).x
    y = landmarks.part(86).y * 1.005
    landmarks_points.append((x, y))
    # 22
    x = landmarks.part(88).x
    y = landmarks.part(88).y
    landmarks_points.append((x, y))
    # 23
    x = landmarks.part(89).x
    y = landmarks.part(89).y * 1.01
    landmarks_points.append((x, y))
    # 24
    x = landmarks.part(90).x
    y = landmarks.part(90).y * 1.02
    landmarks_points.append((x, y))

    # 25
    x = landmarks.part(26).x * 0.97
    y = landmarks.part(26).y * 1.01
    landmarks_points.append((x, y))

    # 26~31
    for n in range(26, 32):
      if n == 26:
        x = landmarks.part(n).x * 0.993
        y = landmarks.part(n).y * 0.995
      elif n == 27:
        x = landmarks.part(n).x
        y = landmarks.part(n).y * 0.995
      else:
        x = landmarks.part(n).x * 1.005
        y = landmarks.part(n).y * 0.995
      landmarks_points.append((x, y))
    # 32~36
    for n in range(33, 38):
      if n == 37:
        x = landmarks.part(n).x * 1.06
        y = landmarks.part(n).y * 1.02
      elif n == 36:
        x = landmarks.part(n).x * 1.03
        y = landmarks.part(n).y * 1.01
      else:
        x = landmarks.part(n).x * 1.007
        y = landmarks.part(n).y * 0.993
      landmarks_points.append((x, y))

    # 37
    x = landmarks.part(91).x
    y = landmarks.part(91).y * 1.07
    landmarks_points.append((x, y))

    #######################################################################

    points = np.array(landmarks_points, np.int32)

    # convexhull 주기
    convexhull = []

    ##########################   left   ################################
    convexhull.append(cv2.convexHull(points[[0, 16, 17]]))
    convexhull.append(cv2.convexHull(points[[0, 1, 14, 15, 16]]))
    convexhull.append(cv2.convexHull(points[[1, 2, 13, 14]]))
    convexhull.append(cv2.convexHull(points[[2, 3, 12, 13]]))
    convexhull.append(cv2.convexHull(points[[3, 4, 11, 12]]))
    convexhull.append(cv2.convexHull(points[[4, 5, 9, 10, 11]]))
    convexhull.append(cv2.convexHull(points[[5, 18, 7, 8, 9]]))
    convexhull.append(cv2.convexHull(points[[18, 6, 7]]))
    ##########################   right   ################################
    convexhull.append(cv2.convexHull(points[[19, 35, 36]]))
    convexhull.append(cv2.convexHull(points[[19, 20, 33, 34, 35]]))
    convexhull.append(cv2.convexHull(points[[20, 21, 32, 33]]))
    convexhull.append(cv2.convexHull(points[[21, 22, 31, 32]]))
    convexhull.append(cv2.convexHull(points[[22, 23, 30, 31]]))
    convexhull.append(cv2.convexHull(points[[23, 24, 28, 29, 30]]))
    convexhull.append(cv2.convexHull(points[[24, 37, 26, 27, 28]]))
    convexhull.append(cv2.convexHull(points[[37, 25, 26]]))

    for con in convexhull:
      cv2.polylines(img, [con], True, (255, 0, 0), 1)
      cv2.fillConvexPoly(eyeshadow_mask, con, 255)

    eyeshadow_image_1 = cv2.bitwise_and(img, img, mask=eyeshadow_mask)

    # Delaunay triangluation
    rect = []
    for con in convexhull:
      rect.append(cv2.boundingRect(con))

    subdiv = []
    triangles = []

    for r in rect:
      subdiv.append(cv2.Subdiv2D(r))

    for i in range(0, len(subdiv)):
      ##########################   left   ################################
      if i == 0:
        subdiv[i].insert(landmarks_points[0])
        subdiv[i].insert(landmarks_points[16:18])
      elif i == 1:
        subdiv[i].insert(landmarks_points[0:2])
        subdiv[i].insert(landmarks_points[14:17])
      elif i == 2:
        subdiv[i].insert(landmarks_points[1:3])
        subdiv[i].insert(landmarks_points[13:15])
      elif i == 3:
        subdiv[i].insert(landmarks_points[2:4])
        subdiv[i].insert(landmarks_points[12:14])
      elif i == 4:
        subdiv[i].insert(landmarks_points[3:5])
        subdiv[i].insert(landmarks_points[11:13])
      elif i == 5:
        subdiv[i].insert(landmarks_points[4:6])
        subdiv[i].insert(landmarks_points[9:12])
      elif i == 6:
        subdiv[i].insert(landmarks_points[5])
        subdiv[i].insert(landmarks_points[18])
        subdiv[i].insert(landmarks_points[7:10])
      elif i == 7:
        subdiv[i].insert(landmarks_points[18])
        subdiv[i].insert(landmarks_points[6:8])
      ##########################   right   ################################
      elif i == 8:
        subdiv[i].insert(landmarks_points[19])
        subdiv[i].insert(landmarks_points[35:37])
      elif i == 9:
        subdiv[i].insert(landmarks_points[19:21])
        subdiv[i].insert(landmarks_points[33:36])
      elif i == 10:
        subdiv[i].insert(landmarks_points[20:22])
        subdiv[i].insert(landmarks_points[32:34])
      elif i == 11:
        subdiv[i].insert(landmarks_points[21:23])
        subdiv[i].insert(landmarks_points[31:33])
      elif i == 12:
        subdiv[i].insert(landmarks_points[22:24])
        subdiv[i].insert(landmarks_points[30:32])
      elif i == 13:
        subdiv[i].insert(landmarks_points[23:25])
        subdiv[i].insert(landmarks_points[28:31])
      elif i == 14:
        subdiv[i].insert(landmarks_points[24])
        subdiv[i].insert(landmarks_points[37])
        subdiv[i].insert(landmarks_points[26:29])
      else:
        subdiv[i].insert(landmarks_points[37])
        subdiv[i].insert(landmarks_points[25:27])

      triangles.append(subdiv[i].getTriangleList())
      triangles[i] = np.array(triangles[i], dtype=np.int32)

    # print(triangles)
    for i in range(0, len(triangles)):
      for t in triangles[i]:
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])

        cv2.line(img, pt1, pt2, (0, 0, 255), 2)
        cv2.line(img, pt2, pt3, (0, 0, 255), 2)
        cv2.line(img, pt1, pt3, (0, 0, 255), 2)

    cv2.imwrite(out_addr, eyeshadow_mask)
    img = Image.open(out_addr)  # 파일 열기
    img = img.convert("RGBA")  # RGBA형식으로 변환
    datas = img.getdata()  # datas에 일차원 배열 형식으로 RGBA입력
    newData = []

    for item in datas:

      if (item[0] == 255 and item[1] == 0 and item[2] == 0):  # 해당 픽셀 색이 흰색이면
        newData.append(item)  # 해당 영역 추가
      else:  # 그렇지 않으면
        newData.append((255, 255, 255, 0))  # 투명 추가

    img.putdata(newData)  # 데이터 입력
    img.save(out_addr)  # 이미지name으로 저장
#
# def Eyeshadow_Extraction(origin_src, out_addr):
#   img = cv2.imread(origin_src)
#
#   img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#   eyeshadow_mask = np.zeros_like(img_gray)
#
#   detector = dlib.get_frontal_face_detector()
#   predictor = dlib.shape_predictor("shape_predictor_194_face_landmarks.dat")
#   faces = detector(img_gray)
#
#   for face in faces:
#     landmarks = predictor(img_gray, face)
#     landmarks_points = []
#
#     for n in range(48, 54):
#       if n == 48:
#         x = landmarks.part(n).x * 1.007
#         y = landmarks.part(n).y * 0.995
#       elif n == 49:
#         x = landmarks.part(n).x
#         y = landmarks.part(n).y * 0.995
#       else:
#         x = landmarks.part(n).x * 0.995
#         y = landmarks.part(n).y * 0.995
#       landmarks_points.append((x, y))



def eyebrow_masking(origin_src, out_addr,center_point,R,G,B):
  img = cv2.imread(origin_src)
  img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  eyebrow_mask = np.zeros_like(img_gray)
  detector = dlib.get_frontal_face_detector()
  predictor = dlib.shape_predictor("shape_predictor_194_face_landmarks.dat")
  faces = detector(img_gray)
  lp=[]
  rp =[]
  max_alpha=20
  for face in faces:
    landmarks = predictor(img_gray, face)
    landmarks_points = []

    # 왼쪽 눈썹
    for n in range(92, 98):
      if (n == 97):
        lx = landmarks.part(n).x
        lp.append(landmarks.part(n).y)
      x = landmarks.part(n).x
      y = landmarks.part(n).y
      landmarks_points.append((x, y))
    for n in range(99, 109):
      if(n == 108):
        lx = landmarks.part(n).x
        lp.append(landmarks.part(n).y)
      x = landmarks.part(n).x
      y = landmarks.part(n).y
      landmarks_points.append((x, y))
    for n in range(110, 114):
      x = landmarks.part(n).x
      y = landmarks.part(n).y
      landmarks_points.append((x, y))

    # 오른쪽 눈썹
    for n in range(70, 76):
      if(n == 75):
        rx = landmarks.part(n).x
        rp.append(landmarks.part(n).y)
      x = landmarks.part(n).x
      y = landmarks.part(n).y
      landmarks_points.append((x, y))
    for n in range(77, 87):
      if (n == 86):
        rx = landmarks.part(n).x
        rp.append(landmarks.part(n).y)
      x = landmarks.part(n).x
      y = landmarks.part(n).y
      landmarks_points.append((x, y))
    for n in range(88, 92):
      x = landmarks.part(n).x
      y = landmarks.part(n).y
      landmarks_points.append((x, y))


    points = np.array(landmarks_points, np.int32)

    convexhull = []
    convexhull.append(cv2.convexHull(points[0:19]))
    convexhull.append(cv2.convexHull(points[20:39]))

    for con in convexhull:
      cv2.polylines(img, [con], True, (255, 0, 0), 1)
      cv2.fillConvexPoly(eyebrow_mask, con, 255)

    eyebrow_image_1 = cv2.bitwise_and(img, img, mask=eyebrow_mask)

    # Delaunay triangluation
    rect = []
    for con in convexhull:
      rect.append(cv2.boundingRect(con))

    subdiv = []
    triangles = []

    for r in rect:
      subdiv.append(cv2.Subdiv2D(r))

    for i in range(0, len(subdiv)):
      if i == 0:
        subdiv[i].insert(landmarks_points[0:18])
      else:
        subdiv[i].insert(landmarks_points[20:38])

      triangles.append(subdiv[i].getTriangleList())
      triangles[i] = np.array(triangles[i], dtype=np.int32)

    # print(triangles)
    for i in range(0, len(triangles)):
      for t in triangles[i]:
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])

        cv2.line(img, pt1, pt2, (0, 0, 255), 2)
        cv2.line(img, pt2, pt3, (0, 0, 255), 2)
        cv2.line(img, pt1, pt3, (0, 0, 255), 2)

  cv2.imwrite(out_addr, eyebrow_mask)

  ly = int((lp[0]+lp[1])/2)
  ry = int((rp[0] + rp[1]) / 2)
  alpha_right = [rx,ry]
  alpha_left = [lx, ly]
  img = Image.open(out_addr)  # 파일 열기
  img = img.convert("RGBA")  # RGBA형식으로 변환
  datas = img.getdata()  # datas에 일차원 배열 형식으로 RGBA입력
  newData = []
  x = -1
  y = 0
  datas = img.getdata()
  newData = []
  for item in datas:
    if (item[0] == 255 and item[1] == 255 and item[2] == 255):  # 해당 픽셀 색이 흰색이면
      newData.append((255,0,0,255))  # 해당 빨강
    else:  # 그렇지 않으면
      newData.append((R, G, B, 0))  # 투명 추가

  img.putdata(newData)  # 데이터 입력
  img.save(out_addr)  # 이미지name으로 저장

  img = Image.open(out_addr)
  opencv_img = cv2.imread(out_addr)
  width = opencv_img.shape[1]
  height = opencv_img.shape[0]
  img = img.convert("RGBA")
  x = -1
  y = 0
  datas = img.getdata()
  newData = []
  min_val, max_val = cal_min_max(out_addr, alpha_right[0], alpha_right[1], alpha_left[0], alpha_left[1], center_point,B,G,R)  # 정규화를 위한 최대 최소 계산
  for item in datas:
    x = x + 1
    if (item[0] == 255 and item[1] == 0 and item[2] == 0 and item[3] == 255 and x < center_point):
      distance = math.sqrt(((alpha_left[0] - x)*0.6) ** 2 + ((alpha_left[1] - y)*4) ** 2)
      newData.append((R, G, B,int(max_alpha - normalization(distance, max_val, min_val) * 100)))  # 거리비례 정규화 역순

    elif (item[0] == 255 and item[1] == 0 and item[2] == 0 and item[3] == 255 and x >= center_point):
      distance = math.sqrt(((alpha_right[0] - x)*0.6) ** 2 + ((alpha_right[1] - y)*4) ** 2)
      newData.append((R, G, B, int(max_alpha - normalization(distance, max_val, min_val) * 100)))
    else:
      newData.append((255, 255, 255, 0))
    if (x >= width):
      y = y + 1
      x = 0
  img.putdata(newData)
  imgname = out_addr
  print("Eyebrow Layer Extract [Path] : " + imgname)
  img.save(imgname, "PNG")


def get_xy_from_landmark(landmarks, n):
  x = landmarks.part(n).x
  y = landmarks.part(n).y
  return (x,y)

def eyeshadow_Extract(origin_src, out_addr,center_point,R,G,B):
  img = cv2.imread(origin_src)
  img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  eyeshadow_mask = np.zeros_like(img_gray)
  max_alpha = 85 #65 default
  detector = dlib.get_frontal_face_detector()
  predictor = dlib.shape_predictor("shape_predictor_194_face_landmarks.dat")
  faces = detector(img_gray)
  right_criteria_len = 0
  left_criteria_len = 0
  weight_X = 1.3
  weight_Y = 2.1
  right_end = (0,0)
  left_end = (0,0)
  right_start=(0,0)
  left_start=(0,0)
  left_eye=[]
  right_eye=[]

  for face in faces:
    landmarks = predictor(img_gray, face)
    get_eyeline(img, faces, landmarks,R,G,B)
    landmarks_points = []
    right_criteria_len = landmarks.part(33).y - landmarks.part(88).y
    left_criteria_len = landmarks.part(57).y-landmarks.part(107).y
    x,y = get_xy_from_landmark(landmarks, 82)
    right_eye.append((x,y))
    x, y = get_xy_from_landmark(landmarks, 83)
    right_eye.append((x, y))
    x, y = get_xy_from_landmark(landmarks, 84)
    right_eye.append((x, y))
    x, y = get_xy_from_landmark(landmarks, 85)
    right_eye.append((x, y))
    x, y = get_xy_from_landmark(landmarks, 86)
    right_eye.append((x, y))
    x, y = get_xy_from_landmark(landmarks, 88)
    right_center=(x,y)
    right_eye.append((x, y))
    x, y = get_xy_from_landmark(landmarks, 89)
    right_eye.append((x, y))
    x, y = get_xy_from_landmark(landmarks, 90)
    right_eye.append((x, y))
    x, y = get_xy_from_landmark(landmarks, 91)
    right_eye.append((x, y)) # 9개 우측 눈썹 포인트 저장
    x, y = get_xy_from_landmark(landmarks, 26) #인덱스 9 ~ 19까지 눈 부분
    right_start=(x, y)
    right_eye.append((x, y))
    x, y = get_xy_from_landmark(landmarks, 27)
    right_eye.append((x, y))
    x, y = get_xy_from_landmark(landmarks, 28)
    right_eye.append((x, y))
    x, y = get_xy_from_landmark(landmarks, 29)
    right_eye.append((x, y))
    x, y = get_xy_from_landmark(landmarks, 30)
    right_eye.append((x, y))
    x, y = get_xy_from_landmark(landmarks, 31)
    right_eye.append((x, y))
    x, y = get_xy_from_landmark(landmarks, 33)
    alpha_right = [x,y]
    right_eye.append((x, y))
    x, y = get_xy_from_landmark(landmarks, 34)
    right_eye.append((x, y))
    x, y = get_xy_from_landmark(landmarks, 35)
    right_eye.append((x, y))
    x, y = get_xy_from_landmark(landmarks, 36)
    right_eye.append((x, y))
    x, y = get_xy_from_landmark(landmarks, 37)
    right_end=(x, y)
    right_eye.append((x, y)) #우측 눈 윗부분 왼쪽부터 11개 입력
    x,y = get_xy_from_landmark(landmarks, 104)
    left_eye.append((x,y))
    x, y = get_xy_from_landmark(landmarks, 105)
    left_eye.append((x, y))
    x, y = get_xy_from_landmark(landmarks, 106)
    left_eye.append((x, y))
    x, y = get_xy_from_landmark(landmarks, 107)
    left_eye.append((x, y))
    x, y = get_xy_from_landmark(landmarks, 108)
    left_eye.append((x, y))
    x, y = get_xy_from_landmark(landmarks, 110)
    left_center = (x, y)
    left_eye.append((x, y))
    x, y = get_xy_from_landmark(landmarks, 111)
    left_eye.append((x, y))
    x, y = get_xy_from_landmark(landmarks, 112)
    left_eye.append((x, y))
    x, y = get_xy_from_landmark(landmarks, 113)
    left_eye.append((x, y)) # 9개 좌측 눈썹 포인트 저장
    x, y = get_xy_from_landmark(landmarks, 48)
    left_start=(x, y)
    left_eye.append((x, y))
    x, y = get_xy_from_landmark(landmarks, 49)
    left_eye.append((x, y))
    x, y = get_xy_from_landmark(landmarks, 50)
    left_eye.append((x, y))
    x, y = get_xy_from_landmark(landmarks, 51)
    left_eye.append((x, y))
    x, y = get_xy_from_landmark(landmarks, 52)
    left_eye.append((x, y))
    x, y = get_xy_from_landmark(landmarks, 53)
    left_eye.append((x, y))
    x, y = get_xy_from_landmark(landmarks, 55)
    alpha_left = [x, y]

    left_eye.append((x, y))
    x, y = get_xy_from_landmark(landmarks, 56)
    left_eye.append((x, y))
    x, y = get_xy_from_landmark(landmarks, 57)
    left_eye.append((x, y))
    x, y = get_xy_from_landmark(landmarks, 58)
    left_eye.append((x, y))
    x, y = get_xy_from_landmark(landmarks, 59)
    left_end=(x, y)
    left_eye.append((x, y)) #좌측 눈 윗부분 오른쪽부터 11개 입력
#        cv2.circle(img, center=tuple(s), radius=1, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
  fill_left=[]
  fill_right=[]

  number = 0
  for i in left_eye:
    temp = i
    if(number==8):
      temp_left_point_x = int(left_start[0] + left_criteria_len * 0.7)
      temp_left_point_y = int(left_start[1] - left_criteria_len * 0.2)
      fill_left.append((temp_left_point_x, temp_left_point_y))
      number = number + 1
      continue
    if(number>=9):
      temp = (i[0],int(i[1]-left_criteria_len*0.2))
    fill_left.append(temp)
    number = number+1

  number = 0
  for j in right_eye:
    if(number == 8):
      temp_right_point_x = int(right_start[0] - right_criteria_len * 0.7)
      temp_right_point_y = int(right_start[1] - right_criteria_len * 0.2)
      fill_right.append((temp_right_point_x, temp_right_point_y))
      number = number + 1
      continue
    temp = j
    if(number>=9):
      temp = (j[0],int(j[1]-right_criteria_len*0.2))

    fill_right.append(temp)
    number = number+1

  temp_left_point_x = int(left_end[0]-left_criteria_len*0.8)
  temp_left_point_y = int(left_end[1]-left_criteria_len*0.1)
  temp_right_point_x = int(right_end[0]+right_criteria_len*0.8)
  temp_right_point_y = int(right_end[1]-right_criteria_len*0.1)

  cv2.circle(eyeshadow_mask,(temp_left_point_x,temp_left_point_y),radius=1, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
  cv2.circle(eyeshadow_mask, (temp_right_point_x, temp_right_point_y), radius=1, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
  fill_left.append((temp_left_point_x,temp_left_point_y))
  fill_right.append((temp_right_point_x, temp_right_point_y))



  fill_left = np.array(fill_left, np.int32)
  fill_right = np.array(fill_right, np.int32)

  eyeshadow_mask = cv2.fillPoly(eyeshadow_mask, [fill_right], (255,255,255))
  eyeshadow_mask = cv2.fillPoly(eyeshadow_mask, [fill_left], (255,255,255))
  cv2.imwrite("test.png", eyeshadow_mask)
  cv2.imwrite(out_addr, eyeshadow_mask)



  img = Image.open(out_addr)
  opencv_img = cv2.imread(out_addr)
  width = opencv_img.shape[1]
  height = opencv_img.shape[0]
  img = img.convert("RGBA")
  x = -1
  y = 0
  datas = img.getdata()
  newData = []
  min_val, max_val = eyebrow_cal_min_max(out_addr, alpha_right[0], alpha_right[1], alpha_left[0], alpha_left[1],center_point)  # 정규화를 위한 최대 최소 계산
  for item in datas:
    x=x+1
    if (item[0]==255 and item[1]== 255 and item[2] == 255 and x <center_point):
      distance = math.sqrt(((alpha_left[0]-x)*weight_X)**2+((alpha_left[1]-y)*weight_Y)**2)
      newData.append((R,G,B,int(max_alpha-normalization(distance,max_val,min_val)*100))) # 거리비례 정규화 역순

    elif(item[0]==255 and item[1]== 255and item[2] == 255 and x >= center_point):
      distance = math.sqrt(((alpha_right[0]-x)*weight_X)**2+((alpha_right[1]-y)*weight_Y)**2)
      newData.append((R,G,B, int(max_alpha-normalization(distance,max_val,min_val)*100)))
    else:
      newData.append((255, 255, 255, 0))
    if (x >= width):
      y = y + 1
      x = 0
  img.putdata(newData)
  imgname = out_addr
  print("Eyeshadow Layer Extract [Path] : " + imgname)
  img.save(imgname, "PNG")

def bitwise_masking(non_bgrimg, max_face):
  #./output/SkinLayer.png
  img = cv2.imread("./output/face_line_img2.0.png")
  origin = cv2.imread(non_bgrimg)
  img2gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  ret,mask = cv2.threshold(img2gray, 254, 255, cv2.THRESH_BINARY)
  cv2.imwrite("mask_before.png", mask)
  mask_inverted = cv2.bitwise_not(mask)
  cv2.imwrite("mask_inverted.png",mask_inverted)
  img1_bg = cv2.bitwise_and(origin, origin, mask = mask)
  width = img1_bg.shape[1]
  height = img1_bg.shape[0]
  cv2.imwrite("cloth.png",img1_bg)
  img = Image.open("cloth.png")  # 파일 열기
  img = img.convert("RGBA")  # RGBA형식으로 변환
  datas = img.getdata()  # datas에 일차원 배열 형식으로 RGBA입력
  newData = []
  x = -1
  y = 0
  for item in datas:
    x = x + 1
    if ((item[0] == 0 and item[1] == 0 and item[2] == 0) or (y<=max_face) or (item[0]==255 and item[1] == 255 and item[2]==255)):  # 해당 픽셀 색이 검정이거나, 턱선 위 일 경우 투명처리
      newData.append((255, 255, 255, 0))
    else:  # 그렇지 않으면
      newData.append(item)  # 해당 영역 추가
    if (x >= width):
      y = y + 1
      x = 0
  img.putdata(newData)  # 데이터 입력
  img.save("cloth.png")  # 이미지name으로 저장

def remove_hair_from_clothes(max_y_from_face):
  mask_cloth = cv2.imread("mask_before.png")
  mask_cloth2gray = cv2.cvtColor(mask_cloth, cv2.COLOR_BGR2GRAY)
  ret, clothmask = cv2.threshold(mask_cloth2gray, 254, 255, cv2.THRESH_BINARY)

  hair = cv2.imread("hair_crop_transparent/crop_hair.png")
  hair2gray = cv2.cvtColor(hair, cv2.COLOR_BGR2GRAY)
  ret, mask = cv2.threshold(hair2gray, 254, 255, cv2.THRESH_BINARY)
  cv2.imwrite("hair_mask_inverted.jpg", mask)



  img1_bg = cv2.bitwise_and(mask_cloth, mask_cloth, mask=mask)
  width = mask_cloth.shape[1]
  height = mask_cloth.shape[0]
  cv2.imwrite("cloth_without_hair.png", img1_bg)

  color_pattern = Image.open("cloth_pattern.png")
  color_pattern = color_pattern.convert("RGBA")

  without_hair = Image.open("cloth_without_hair.png")
  without_hair = without_hair.convert("RGBA")

  color_pattern_data = color_pattern.getdata()
  without_hairdata = without_hair.getdata()

  newData = []
  x = -1
  y = 0
  for i in range(len(color_pattern_data)):
    x = x + 1
    if ((color_pattern_data[i][0] == 255 and color_pattern_data[i][1] == 255 and color_pattern_data[i][2] == 255 and color_pattern_data[i][3] == 0)):
      newData.append((255, 255, 255, 0))
    elif(y>max_y_from_face*3):
      newData.append(color_pattern_data[i])
    elif((without_hairdata[i][0] == 0 and without_hairdata[i][1] == 0 and without_hairdata[i][2] == 0)):
      newData.append((255, 255, 255, 0))
    else:
      newData.append(color_pattern_data[i])
    if (x >= width):
      y = y + 1
      x = 0
  without_hair.putdata(newData)  # 데이터 입력
  without_hair.save("cloth_without_hair_pattern.png")

def get_eyeline(img, faces, landmarks,R,G,B):
  img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  left_eye = []
  right_eye = []
  calculation=[]
  for face in faces:
    x, y = get_xy_from_landmark(landmarks, 48)
    calculation.append((x,y))
    x, y = get_xy_from_landmark(landmarks, 49)
    calculation.append((x,y))

    move_up_pixel = int(abs(calculation[1][1]-calculation[0][1])/1.8)
    move_side_pixel = int(abs(calculation[1][0] - calculation[0][0]) / 1.5)

    x, y = get_xy_from_landmark(landmarks, 48)
    left_eye.append((x, y-move_up_pixel))
    x, y = get_xy_from_landmark(landmarks, 49)
    left_eye.append((x, y-move_up_pixel))
    x, y = get_xy_from_landmark(landmarks, 51)
    left_eye.append((x, y-move_up_pixel))
    x, y = get_xy_from_landmark(landmarks, 52)
    left_eye.append((x, y-move_up_pixel))
    x, y = get_xy_from_landmark(landmarks, 53)
    left_eye.append((x, y-move_up_pixel))
    x, y = get_xy_from_landmark(landmarks, 55)
    left_eye.append((x, y-move_up_pixel))
    x, y = get_xy_from_landmark(landmarks, 56)
    left_eye.append((x-move_side_pixel, y-move_up_pixel))
    x, y = get_xy_from_landmark(landmarks, 57)
    left_eye.append((x-move_side_pixel, y-move_up_pixel))
    move_up_pixel = int(abs(calculation[1][1]-calculation[0][1])/1.6)
    move_side_pixel = int(abs(calculation[1][0] - calculation[0][0]) / 1)
    x, y = get_xy_from_landmark(landmarks, 58)
    left_eye.append((x-move_side_pixel, y-move_up_pixel))
    x, y = get_xy_from_landmark(landmarks, 59)
    left_eye.append((x-move_side_pixel, y-move_up_pixel))
    left_width = abs(left_eye[-1][0]-left_eye[-2][0])
    left_height = abs(left_eye[-1][1]-left_eye[-2][1])
    move_side_pixel = int(abs(calculation[1][0] - calculation[0][0]) / 1.5)
    move_up_pixel = int(abs(calculation[1][1]-calculation[0][1])/1.8)
    x, y = get_xy_from_landmark(landmarks, 26)
    right_eye.append((x, y-move_up_pixel))
    x, y = get_xy_from_landmark(landmarks, 27)
    right_eye.append((x, y-move_up_pixel))
    x, y = get_xy_from_landmark(landmarks, 28)
    right_eye.append((x, y-move_up_pixel))
    x, y = get_xy_from_landmark(landmarks, 29)
    right_eye.append((x, y-move_up_pixel))
    x, y = get_xy_from_landmark(landmarks, 30)
    right_eye.append((x, y-move_up_pixel))
    x, y = get_xy_from_landmark(landmarks, 31)
    right_eye.append((x, y-move_up_pixel))
    x, y = get_xy_from_landmark(landmarks, 33)
    right_eye.append((x, y-move_up_pixel))
    x, y = get_xy_from_landmark(landmarks, 34)
    right_eye.append((x+move_side_pixel, y-move_up_pixel))
    x, y = get_xy_from_landmark(landmarks, 35)
    right_eye.append((x+move_side_pixel, y-move_up_pixel))
    x, y = get_xy_from_landmark(landmarks, 36)
    move_side_pixel = int(abs(calculation[1][0] - calculation[0][0]) / 1)
    right_eye.append((x+move_side_pixel, y-move_up_pixel))
    x, y = get_xy_from_landmark(landmarks, 37)
    right_eye.append((x+move_side_pixel, y-move_up_pixel))

    right_width = abs(right_eye[-1][0] - right_eye[-2][0])
    right_height =  abs(right_eye[-1][1] - right_eye[-2][1])

    left_eye.append((left_eye[-1][0]-int(left_width/2), left_eye[-1][1] + int(left_height/2)))
    right_eye.append((right_eye[-1][0] + int(right_width/2), right_eye[-1][1] + int(right_height/2)))
    left_eye.append((left_eye[-1][0] - int(left_width / 4), left_eye[-1][1] + int(left_height / 4)))
    right_eye.append((right_eye[-1][0] + int(right_width / 4), right_eye[-1][1] + int(right_height / 4)))

    width = img.shape[1]
    height = img.shape[0]

    for i in range(len(left_eye)-1):
      if(i>=9):
        cv2.line(img, (left_eye[i][0], left_eye[i][1]), (left_eye[i + 1][0], left_eye[i + 1][1]), color=(0, 0, 255),
                 thickness=int(right_height / 2), lineType=cv2.LINE_AA)
      else:
        cv2.line(img, (left_eye[i][0], left_eye[i][1]), (left_eye[i + 1][0], left_eye[i + 1][1]), color=(0, 0, 255),
                 thickness=int(right_height / 1), lineType=cv2.LINE_AA)

    for i in range(len(right_eye)-1):
      if(i>=10):
        print(i)
        cv2.line(img, (right_eye[i][0],right_eye[i][1]), (right_eye[i+1][0],right_eye[i+1][1]), color = (0,0,255), thickness = int(right_height / 2), lineType=cv2.LINE_AA)
      else:
        cv2.line(img, (right_eye[i][0],right_eye[i][1]), (right_eye[i+1][0],right_eye[i+1][1]), color = (0,0,255), thickness = int(right_height / 1), lineType=cv2.LINE_AA)

    cv2.imwrite("output/eyeline.png", img)

    img = Image.open("output/eyeline.png")
    img = img.convert("RGBA")
    x = -1
    y = 0
    datas = img.getdata()
    newData = []
    for item in datas:
      x = x + 1
      if (item[0] == 255 and item[1] == 0 and item[2] == 0 and item[3]==255):
        newData.append((R, G, B, 130))
      else:
        newData.append((255, 255, 255, 0))
      if (x >= width):
        y = y + 1
        x = 0
    img.putdata(newData)
    imgname = "output/eyeline.png"
    print("Eyeline Layer Extract [Path] : " + imgname)
    img.save(imgname, "PNG")




def accumulate_hair_layer():
  layer1 = Image.open("./hair_crop_transparent/crop_hair.png").convert("RGBA")
  #layer2 = Image.open("./hair_crop_transparent/crop_hair1.png").convert("RGBA")
  #layer3 = Image.open("./hair_crop_transparent/crop_hair2.png").convert("RGBA")
  layer4 = Image.open("./hair_crop_transparent/crop_hair3.png").convert("RGBA")
  #layer5 = Image.open("./hair_crop_transparent/crop_hair4.png").convert("RGBA")
  #layer6 = Image.open("./hair_crop_transparent/crop_hair5.png").convert("RGBA")

  #result = Image.alpha_composite(layer1, layer2)
  #result = Image.alpha_composite(result, layer3)
  #result = Image.alpha_composite(result, layer4)
  #result = Image.alpha_composite(result, layer5)
  #result = Image.alpha_composite(result, layer6)

  layer1.save("./hair_crop_transparent/crop_hair.png")