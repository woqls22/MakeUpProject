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
  print("Extraction : Nose Part")
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
  print("Extraction : Eye Part")
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
  print("Extraction : Face Part")

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
  if(item[0]<item[1] or item[0]<item[2]):
    return False
  if(item[0]>item[1]+120 ):
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
      x=x+1
      if(is_face(item) or (x>minX and x<maxX)):
        if(is_face(item)):
          newData.append(item)
        else:
          newData.append((255, 255, 255, 0))
          pass
      else:
        newData.append((255, 255, 255, 0))
    if (x >= width):
        y = y + 1
        x=0
    img.putdata(newData)
    img.save(imgname, "PNG")

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
    cv2.imwrite('./output/face_line_img'+version+'.png', cvImg)
    #cv2.imshow('test', cvImg)



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
  imgn = "./output/CheekLayer.png"
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
    imgname = "./output/CheekLayer.png"
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

def get_lip_layer(imgname, mouse):
  x = -1
  y = 0
  x_list=[]
  y_list=[]
  for i in mouse:
    x_list.append(i[0])
    y_list.append(i[1])
  min_x = min(x_list)-5
  max_x = max(x_list)+5
  min_y = min(y_list)-3
  max_y = max(y_list)+5
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
    center_x = int((min_x+max_x)/2)
    center_y = int((min_y + max_y) / 2)
    centerLip_color = opencv_img[center_y,center_x]
    for item in datas:
      if(is_in_Area(x,y,min_x,max_x,min_y,max_y) and Lip_Similar_with_point(centerLip_color, item)):
        newData.append(item)
      else:
        newData.append((255,255,255,0))
      x = x + 1
      if (x >= width):
        y = y + 1
        x = 0

    img.putdata(newData)
    imgname = "./output/lip_layer.png"
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
        newData.append(item)
      elif((is_in_Area(x, y, righteyebrowmin_x, righteyebrowmax_x, righteyebrowmin_y, righteyebrowmax_y) and (Eyebrow_Similar_with_point(right_center_eyebrow_color, item)))):
        newData.append(item)
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
   print(source_color)
   if(obj_b>170 or obj_g>170):
     return False
   if(abs(sorce_r - obj_r)<red_confidence and abs(sorce_g - obj_g)<confidence and abs(sorce_b - obj_b)<confidence):
     return True
   if(abs(sorce_r - obj_g)>60 and abs(sorce_r - obj_b)>60):
     return True
   return False
def Eyebrow_Similar_with_point(source_color, compare_obj):
  red_confidence = 23
  confidence = 100
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
  return False

def eyemasking(origin_src, out_addr):
  img = cv2.imread(origin_src)
  img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  eye_mask = np.zeros_like(img_gray)

  detector = dlib.get_frontal_face_detector()
  predictor = dlib.shape_predictor("shape_predictor_194_face_landmarks.dat")
  faces = detector(img_gray)
  for face in faces:
    landmarks = predictor(img_gray, face)
    landmarks_points = []

    # 왼쪽 눈
    for n in range(48, 54):
      x = landmarks.part(n).x
      y = landmarks.part(n).y * 0.98
      landmarks_points.append((x, y))
    for n in range(55, 65):
      x = landmarks.part(n).x
      y = landmarks.part(n).y * 0.98
      landmarks_points.append((x, y))
    for n in range(66, 70):
      x = landmarks.part(n).x
      y = landmarks.part(n).y * 0.98
      landmarks_points.append((x, y))

    # 오른쪽 눈
    for n in range(26, 32):
      x = landmarks.part(n).x
      y = landmarks.part(n).y * 0.98
      landmarks_points.append((x, y))
    for n in range(33, 43):
      x = landmarks.part(n).x
      y = landmarks.part(n).y * 0.98
      landmarks_points.append((x, y))
    for n in range(44, 48):
      x = landmarks.part(n).x
      y = landmarks.part(n).y * 0.98
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

      if (item[0] == 255 and item[1] == 255 and item[2] == 255):  # 해당 픽셀 색이 흰색이면
        newData.append(item)  # 해당 영역 추가
      else:  # 그렇지 않으면
        newData.append((255, 255, 255, 0))  # 투명 추가

    img.putdata(newData)  # 데이터 입력
    img.save(out_addr)  # 이미지name으로 저장

def eyebrow_masking(origin_src, out_addr):
  img = cv2.imread(origin_src)
  img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  eyebrow_mask = np.zeros_like(img_gray)

  detector = dlib.get_frontal_face_detector()
  predictor = dlib.shape_predictor("shape_predictor_194_face_landmarks.dat")
  faces = detector(img_gray)
  for face in faces:
    landmarks = predictor(img_gray, face)
    landmarks_points = []

    # 왼쪽 눈썹
    for n in range(92, 98):
      x = landmarks.part(n).x
      y = landmarks.part(n).y
      landmarks_points.append((x, y))
    for n in range(99, 109):
      x = landmarks.part(n).x
      y = landmarks.part(n).y
      landmarks_points.append((x, y))
    for n in range(110, 114):
      x = landmarks.part(n).x
      y = landmarks.part(n).y
      landmarks_points.append((x, y))

    # 오른쪽 눈썹
    for n in range(70, 76):
      x = landmarks.part(n).x
      y = landmarks.part(n).y
      landmarks_points.append((x, y))
    for n in range(77, 87):
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