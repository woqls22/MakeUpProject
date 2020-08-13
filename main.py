import Util as U
import cv2
import dlib
import numpy as np
import cv2
from matplotlib import pyplot as plt
import pattern
from PIL import Image
import os


# Picture FileName
# Illust Flag => True이면 Cartoon GAN 적용 후 Face Layer 추출, False이면 미적용, 추출
convert_to_illust = False
Remove_BG=True
Get_HairSeg=False
R=255
G=0
B=0
# color setting

FileName = "mental.png"

input_file=""
if(Remove_BG):
  import remove_bgr as RB
  img_Name,input_file = RB.start()
  FileName = img_Name.split('/')[2]
else:
  print("Enter the File Name : ", end='')
  FileName = input()
  input_file=FileName
print("FileName : "+FileName)
ImgName = './input/'+FileName
img_Name = ImgName
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
version="2.0"
#light_paprika_ckpt, light_shinkai_ckpt
modelpath = "./light_shinkai_ckpt"
img_path = ImgName
out_dir = './output/'
U.opened_mouse_layer(img_Name, R,G,B)
pattern_file = ["pattern1.png","pattern2.png","pattern3.png","pattern4.png","pattern5.png","pattern6.jpg"]

print("Pattern List : ",end='')
print(pattern_file)
if(convert_to_illust):
  import inference_with_ckpt as illustrator
  illustrator.convert(modelpath, img_path,out_dir)
  ImgName = out_dir+FileName
face_roi = []
face_sizes = []
U.eyemasking(img_path, './output/eye'+version+'.png')
U.eyeshadow_masking(img_path, './output/eyeshadow'+version+'.png')
while True:
  img = cv2.imread(ImgName)
  print("Input Image : "+ImgName)
  ori = img.copy()
  MS_ori = img.copy()
  LE_ori = img.copy()
  RE_ori = img.copy()
  NO_ori = img.copy()
  FL_ori = img.copy()
  LEB_ori = img.copy()
  REB_ori = img.copy()
  cheek_ori = img.copy()

  # find faces
  if len(face_roi) == 0:
    faces = detector(img, 1)
  else:
    roi_img = img[face_roi[0]:face_roi[1], face_roi[2]:face_roi[3]]
    # cv2.imshow('roi', roi_img)
    faces = detector(roi_img)

  # find facial landmarks
  for face in faces:
    if len(face_roi) == 0:
      dlib_shape = predictor(img, face)
      shape_2d = np.array([[p.x, p.y] for p in dlib_shape.parts()])
    else:
      dlib_shape = predictor(roi_img, face)
      shape_2d = np.array([[p.x + face_roi[2], p.y + face_roi[0]] for p in dlib_shape.parts()])
    point_number = 0
    mouse=[]
    left_eyes=[]
    right_eyes=[]
    nose = []
    face_line = []
    left_eyes_brow=[]
    right_eyes_brow=[]
    points_for_leftcheek = []
    points_for_rightcheek = []
    points_for_nosecheek=[]
    end_of_face = 0
    start_of_face=0
    max_y_from_face = 0
    for s in shape_2d:
      ## 얼굴 각 부위를 Array에 넣음.
      if(point_number>=48 and point_number<=59): ## Mouse Part
        cv2.circle(img, center=tuple(s), radius=1, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        mouse.append(s)
        if(point_number==49): #left cheek points
          cv2.circle(img, center=tuple(s), radius=1, color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
          points_for_leftcheek.append(s)

        if(point_number==53): #right cheek points
          cv2.circle(img, center=tuple(s), radius=1, color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
          points_for_rightcheek.append(s)


      elif(point_number>=36and point_number<=39): #left Abobe Part
        cv2.circle(img, center=tuple(s), radius=1, color=(255, 255, 0), thickness=2, lineType=cv2.LINE_AA)

        if (point_number == 39):  # left cheek points
          cv2.circle(img, center=tuple(s), radius=1, color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
          points_for_leftcheek.append(s)
        left_eyes.append(s)
      elif (point_number >= 40 and point_number <= 41):  # left_eyes below Part
        cv2.circle(img, center=tuple(s), radius=1, color=(255, 255, 0), thickness=2, lineType=cv2.LINE_AA)
        left_eyes.append(s)

      elif (point_number >= 42 and point_number <= 45):  # righteyes Abobe Part
        cv2.circle(img, center=tuple(s), radius=1, color=(30, 255, 54), thickness=2, lineType=cv2.LINE_AA)
        if (point_number == 42):  # right cheek points
          cv2.circle(img, center=tuple(s), radius=1, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
          points_for_rightcheek.append(s)
        right_eyes.append(s)

      elif (point_number >= 46 and point_number <= 47):  # righteyes Below Part
        cv2.circle(img, center=tuple(s), radius=1, color=(30, 255, 54), thickness=2, lineType=cv2.LINE_AA)
        right_eyes.append(s)
      elif (point_number == 27):  # nose Part
        cv2.circle(img, center=tuple(s), radius=1, color=(140, 0, 255), thickness=2, lineType=cv2.LINE_AA)
        points_for_nosecheek.append(s)
        nose.append(s)
      elif(point_number>=28 and point_number<=30): # nose Part
        cv2.circle(img, center=tuple(s), radius=1, color=(140, 0, 255), thickness=2, lineType=cv2.LINE_AA)
        nose.append(s)
      elif(point_number>=31 and point_number<=35): # nose Part
        cv2.circle(img, center=tuple(s), radius=1, color=(140, 0, 255), thickness=2, lineType=cv2.LINE_AA)
        nose.append(s)
      elif(point_number>=0 and point_number<=16):#face_line Part
        cv2.circle(img, center=tuple(s), radius=1, color=(175, 255, 130), thickness=2, lineType=cv2.LINE_AA)
        face_line.append(s)
        if(point_number==1  or point_number==4): #왼쪽뺨 볼터치 레이어 point
          cv2.circle(img, center=tuple(s), radius=1, color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
          points_for_leftcheek.append(s)
        if(point_number==12 or point_number==15): #오른쪽 뺨 볼터치 레이어 point
          cv2.circle(img, center=tuple(s), radius=1, color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
          points_for_rightcheek.append(s)
        if(point_number==5):
          end_of_face=s[1]
        if(point_number==9):
          max_y_from_face=s[1]
      elif (point_number >= 17 and point_number <= 21):  # left_eyes_brow Part
        cv2.circle(img, center=tuple(s), radius=1, color=(170, 170, 170), thickness=2, lineType=cv2.LINE_AA)
        left_eyes_brow.append(s)

      elif (point_number >= 22 and point_number <= 26):  # right_eyes_brow Part
        cv2.circle(img, center=tuple(s), radius=1, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
        right_eyes_brow.append(s)

      else:
        cv2.circle(img, center=tuple(s), radius=1, color=(0, 0, 0), thickness=2, lineType=cv2.LINE_AA)

      point_number = point_number+1
    #U.draw_line(img, list(face_line))
    # compute face center
    center_x, center_y = np.mean(shape_2d, axis=0).astype(np.int)
    # 눈썹, 눈, 입 제거
    U.erase_layer(FL_ori,version, left_eyes_brow, right_eyes_brow, left_eyes, right_eyes, mouse)
    # compute face boundaries
    min_coords = np.min(shape_2d, axis=0)
    max_coords = np.max(shape_2d, axis=0)

    # draw min, max coords
    cv2.circle(img, center=tuple(min_coords), radius=1, color=(255, 0, 0), thickness=1, lineType=cv2.LINE_AA)
    cv2.circle(img, center=tuple(max_coords), radius=1, color=(255, 0, 0), thickness=1, lineType=cv2.LINE_AA)

    # compute face size
    face_size = max(max_coords - min_coords)
    face_sizes.append(face_size)
    if len(face_sizes) > 10:
      del face_sizes[0]
    mean_face_size = int(np.mean(face_sizes) * 1.8)

    # compute face roi
    face_roi = np.array([int(min_coords[1] - face_size / 2), int(max_coords[1] + face_size / 2), int(min_coords[0] - face_size / 2), int(max_coords[0] + face_size / 2)])
    face_roi = np.clip(face_roi, 0, 10000)
    # 이미지별 저장
    mouse_img = U.extract_part(MS_ori,mouse)
    left_eyes_img = U.extract_eye_part(LE_ori,left_eyes)
    right_eyes_img = U.extract_eye_part(RE_ori, right_eyes)
    nose_img = U.extract_nose_part(NO_ori, nose)
    face_line_img = U.extract_face_part(FL_ori, face_line)
    left_eyes_brow_img = U.extract_part(LEB_ori, left_eyes_brow)
    right_eyes_brow_img = U.extract_part(REB_ori, right_eyes_brow)

  # visualize
  U.get_cheek_layer(cheek_ori, points_for_leftcheek, points_for_rightcheek, face_line,points_for_nosecheek[0][0],R,G,B)
  cv2.imwrite(out_dir+'cheek_colored'+version+'.png',cheek_ori)
  # Img Write
  cv2.imwrite(out_dir+'mouse'+version+'.png',mouse_img)
  cv2.imwrite(out_dir+'left_eyes_img'+version+'.png', left_eyes_img )
  cv2.imwrite(out_dir+'right_eyes_img'+version+'.png', right_eyes_img )
  cv2.imwrite(out_dir+'nose_img'+version+'.png', nose_img )
  cv2.imwrite(out_dir+'face_line_img'+version+'.png', face_line_img)
  U.eyebrow_masking(img_path, './output/eyebrow' + version + '.png',points_for_nosecheek[0][0],R,G,B)

  cv2.imwrite(out_dir+'left_eyes_brow_img'+version+'.png', left_eyes_brow_img )
  cv2.imwrite(out_dir+'right_eyes_brow_img'+version+'.png', right_eyes_brow_img )
  U.eyeshadow_Extract(img_path, './output/eyeshadow_modified' + version + '.png', points_for_nosecheek[0][0],R,G,B)
  print("mouse Layer Extract [Path] : " + out_dir+'mouse'+version+'.png')
  print("left_eyes_img Layer Extract [Path] : " + out_dir+'left_eyes_img'+version+'.png')
  print("right_eyes_img Layer Extract [Path] : " + out_dir+'right_eyes_img'+version+'.png')
  print("nose_img Layer Extract [Path] : " + out_dir+'nose_img'+version+'.png')
  print("face_line_img Layer Extract [Path] : " + out_dir+'face_line_img'+version+'.png')
  lip_fname = out_dir+'mouse'+version+'.png'
  U.get_lip_layer(ImgName,mouse,R,G,B)

  U.get_eyebrow_layer(ImgName, left_eyes_brow, right_eyes_brow)


  x = []
  y = []
  for i in range(len(face_line)):
    x.append(face_line[0:len(face_line)][i][0])
    y.append(face_line[0:len(face_line)][i][1])
  minX = x[5]
  maxX = x[11]
  minY = min(y)
  maxY = 0
  nose_x= []
  nose_y = []
  for i in range(len(nose)):
    nose_x.append(nose[0:len(nose)][i][0])
    nose_y.append(nose[0:len(nose)][i][1])

  U.get_rid_of_face_background(out_dir+'face_line_img'+version+'.png', minX,maxX,end_of_face)
  U.postprocess_face_layer(out_dir+'face_line_img'+version+'.png', nose_x, nose_y)

  face_cover = cv2.imread(out_dir+'face_line_img'+version+'.png')

  print("=====")
  print("[Extraction] Each Face Part Layer Complete .... ")
  U.bitwise_masking(img_Name,end_of_face)

  original = "./input/"+input_file
  for pattern_source in pattern_file:
    pattern.get_pattern(pattern_source)

  if (Get_HairSeg):
    import hair_segmentation as HS
    os.environ["KERAS_BACKEND"] = "tensorflow"
    HS.hair_segment(original) #hair segmentation작업
    #U.accumulate_hair_layer() #레이어 쌓기
    U.remove_hair_from_clothes(max_y_from_face) # 마스킹작업
    cloth_layer = "./cloth_without_hair_pattern.png"
  else:
    cloth_layer = "./cloth_pattern1.png"

  print("[Extraction] Cloth Layer Complete .... ")
  print("=====")
  print("Process Done.")


  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

  cheek_layer = "./output/CheekLayer.png"
  eyeshadow_layer = "./output/eyeshadow_modified2.0.png"
  eyebrow_layer = "./output/eyebrow2.0.png"
  lip_layer = "./output/lip_layer.png"



  layer1 = Image.open(original).convert("RGBA")
  layer2 = Image.open(cheek_layer).convert("RGBA")
  layer3 = Image.open(eyeshadow_layer).convert("RGBA")
  layer4 = Image.open(eyebrow_layer).convert("RGBA")
  layer5 = Image.open(lip_layer).convert("RGBA")

  layer6 = Image.open(cloth_layer).convert("RGBA")
  layer7 = Image.open("./output/eyeline.png").convert("RGBA")


  Makeup_result = Image.alpha_composite(layer1, layer2)
  Makeup_result = Image.alpha_composite(Makeup_result, layer3)
  Makeup_result = Image.alpha_composite(Makeup_result, layer4)
  Makeup_result = Image.alpha_composite(Makeup_result, layer5) #레이어 쌓아 올리기
  Makeup_result = Image.alpha_composite(Makeup_result, layer7)
  Makeup_result.save("result/Makeup_result.png")  # 이미지name으로 저장



  SwapClothes = Image.alpha_composite(layer1, layer6) #옷 입힘
  SwapClothes = Image.alpha_composite(SwapClothes, layer2) #cheek_layer
  SwapClothes = Image.alpha_composite(SwapClothes, layer3)  # eyeshadow_layer
  SwapClothes = Image.alpha_composite(SwapClothes, layer4)  # eyebrow_layer
  SwapClothes = Image.alpha_composite(SwapClothes, layer5)  # lip_layer
  SwapClothes.save("result/SwapClothes.png")

  fig = plt.figure()
  rows = 1
  cols = 3
  ax1 = fig.add_subplot(rows, cols, 1)
  ax1.imshow(layer1)
  ax1.set_xlabel('Before')
  ax1.set_xticks([]), ax1.set_yticks([])

  ax2 = fig.add_subplot(rows, cols, 2)
  ax2.imshow(Makeup_result)
  ax2.set_xlabel('After')
  ax2.set_xticks([]), ax2.set_yticks([])

  ax2 = fig.add_subplot(rows, cols, 3)
  ax2.imshow(SwapClothes)
  ax2.set_xlabel('Swap Clothes')
  ax2.set_xticks([]), ax2.set_yticks([])

  plt.show()
  cv2.waitKey(0)
  break