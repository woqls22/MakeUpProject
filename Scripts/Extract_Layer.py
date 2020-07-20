import cv2, dlib, sys
import numpy as np
from PIL import Image
import Util as U
scaler = 0.5

# initialize face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
version="1.0"

# load video
#cap = cv2.VideoCapture('./sample.png')
# load overlay image
#overlay = cv2.imread('samples/ryan_transparent.png', cv2.IMREAD_UNCHANGED)


face_roi = []
face_sizes = []
# loop
while True:
  # read frame buffer from video
  #ret, img = cap.read()
  #if not ret:
  #break
  img = cv2.imread('./test.png')
  # resize frame
  #img = cv2.resize(img, (int(img.shape[1] * scaler), int(img.shape[0] * scaler)))
  ori = img.copy()
  MS_ori = img.copy()
  LE_ori = img.copy()
  RE_ori = img.copy()
  NO_ori = img.copy()
  FL_ori = img.copy()
  LEB_ori = img.copy()
  REB_ori = img.copy()


  # find faces
  if len(face_roi) == 0:
    faces = detector(img, 1)
  else:
    roi_img = img[face_roi[0]:face_roi[1], face_roi[2]:face_roi[3]]
    # cv2.imshow('roi', roi_img)
    faces = detector(roi_img)

  # no faces


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
    for s in shape_2d:
      ## 얼굴 각 부위를 Array에 넣음.
      if(point_number>=48): ## Mouse Part
        cv2.circle(img, center=tuple(s), radius=1, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        mouse.append(s)
      elif(point_number>=36and point_number<=41): #right_eyes Part
        cv2.circle(img, center=tuple(s), radius=1, color=(255, 255, 0), thickness=2, lineType=cv2.LINE_AA)
        right_eyes.append(s)
      elif (point_number >= 42 and point_number <= 47):  # left_eyes Part
        cv2.circle(img, center=tuple(s), radius=1, color=(30, 255, 54), thickness=2, lineType=cv2.LINE_AA)
        left_eyes.append(s)
      elif(point_number>=27 and point_number<=35): # nose Part
        cv2.circle(img, center=tuple(s), radius=1, color=(140, 0, 255), thickness=2, lineType=cv2.LINE_AA)
        nose.append(s)
      elif(point_number>=0 and point_number<=16):#face_line Part
        cv2.circle(img, center=tuple(s), radius=1, color=(175, 255, 130), thickness=2, lineType=cv2.LINE_AA)
        face_line.append(s)
      elif (point_number >= 17 and point_number <= 21):  # left_eyes_brow Part
        cv2.circle(img, center=tuple(s), radius=1, color=(170, 170, 170), thickness=2, lineType=cv2.LINE_AA)
        left_eyes_brow.append(s)
      elif (point_number >= 22 and point_number <= 26):  # right_eyes_brow Part
        cv2.circle(img, center=tuple(s), radius=1, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
        right_eyes_brow.append(s)
      else:
        print("EXCEPT")
      point_number = point_number+1
    #draw_line(ori, list(mouse))
    # compute face center
    center_x, center_y = np.mean(shape_2d, axis=0).astype(np.int)

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

    mouse_img = U.extract_part(MS_ori,mouse)
    left_eyes_img = U.extract_eye_part(LE_ori,left_eyes)
    right_eyes_img = U.extract_eye_part(RE_ori, right_eyes)
    nose_img = U.extract_nose_part(NO_ori, nose)
    face_line_img = U.extract_face_part(FL_ori, face_line)
    left_eyes_brow_img = U.extract_part(LEB_ori, left_eyes_brow)
    right_eyes_brow_img = U.extract_part(REB_ori, right_eyes_brow)
  # visualize
  cv2.imshow('original', ori)
  cv2.imshow('facial landmarks', img)
  cv2.imwrite('./FacePart/mouse'+version+'.png',mouse_img )
  cv2.imwrite('./FacePart/left_eyes_img'+version+'.png', left_eyes_img )
  cv2.imwrite('./FacePart/right_eyes_img'+version+'.png', right_eyes_img )
  cv2.imwrite('./FacePart/nose_img'+version+'.png', nose_img )
  cv2.imwrite('./FacePart/face_line_img'+version+'.png', face_line_img )
  cv2.imwrite('./FacePart/left_eyes_brow_img'+version+'.png', left_eyes_brow_img )
  cv2.imwrite('./FacePart/right_eyes_brow_img'+version+'.png', right_eyes_brow_img )
  #def get_face_layer(imgname, mouse, left_eyes,right_eyes,nose,face_line,left_eyes_brow,right_eyes_brow):
  #get_face_layer('./FacePart/face_line_img.png', mouse, left_eyes, right_eyes, nose, face_line, left_eyes_brow, right_eyes_brow)
  U.get_rid_of_background('./FacePart/mouse'+version+'.png')
  U.get_rid_of_background('./FacePart/left_eyes_img'+version+'.png')
  U.get_rid_of_background('./FacePart/right_eyes_img'+version+'.png')
  U.get_rid_of_background('./FacePart/nose_img'+version+'.png')
  U.get_rid_of_face_background('./FacePart/face_line_img'+version+'.png')
  U.get_rid_of_background('./FacePart/left_eyes_brow_img'+version+'.png')
  U.get_rid_of_background('./FacePart/right_eyes_brow_img'+version+'.png')

  print("Extraction Face Part Complete")
  #cv2.imwrite('q./result.png',img)

  cv2.waitKey(0)
  #if cv2.waitKey(1) == ord('q'):
  #sys.exit(1)
  cv2.destroyAllWindows()
  break