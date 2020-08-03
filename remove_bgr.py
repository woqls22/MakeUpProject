import os
from io import BytesIO

import numpy as np
from PIL import Image

import tensorflow as tf
import sys
import datetime
import cv2

class DeepLabModel(object):
  """Class to load deeplab model and run inference."""

  INPUT_TENSOR_NAME = 'ImageTensor:0'
  OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
  INPUT_SIZE = 513
  FROZEN_GRAPH_NAME = 'frozen_inference_graph'

  def __init__(self, tarball_path):
    """Creates and loads pretrained deeplab model."""
    self.graph = tf.Graph()

    graph_def = None
    graph_def = tf.GraphDef.FromString(open(tarball_path + "/frozen_inference_graph.pb", "rb").read()) 

    if graph_def is None:
      raise RuntimeError('Cannot find inference graph in tar archive.')

    with self.graph.as_default():
      tf.import_graph_def(graph_def, name='')

    self.sess = tf.Session(graph=self.graph)

  def run(self, image):
    """Runs inference on a single image.

    Args:
      image: A PIL.Image object, raw input image.

    Returns:
      resized_image: RGB image resized from original input image.
      seg_map: Segmentation map of `resized_image`.
    """
    start = datetime.datetime.now()

    width, height = image.size
    resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
    target_size = (int(resize_ratio * width), int(resize_ratio * height))
    resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
    batch_seg_map = self.sess.run(
        self.OUTPUT_TENSOR_NAME,
        feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
    seg_map = batch_seg_map[0]

    end = datetime.datetime.now()

    diff = end - start
    print("Time taken to evaluate segmentation is : " + str(diff))

    return resized_image, seg_map

def drawSegment(baseImg, matImg,inputFilePath,outputFilePath):
  width, height = baseImg.size
  origin = cv2.imread(inputFilePath)
  height1,width1,channel = origin.shape
  dummyImg = np.zeros([height, width, 4], dtype=np.uint8)
  for x in range(width):
            for y in range(height):
                color = matImg[y,x]
                (r,g,b) = baseImg.getpixel((x,y))
                if color == 0:
                    dummyImg[y,x,3] = 0
                else :
                    dummyImg[y,x] = [r,g,b,255]
  img = Image.fromarray(dummyImg)
  img.save(outputFilePath)
  frame = cv2.imread(outputFilePath)
  frame = cv2.resize(frame, (width1,height1))
  frame2gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  ret,mask = cv2.threshold(frame2gray, 0, 255, cv2.THRESH_BINARY)
  cv2.imwrite("mask.jpg",mask)

  img1_bg = cv2.bitwise_or(origin, origin, mask = mask)
  img2_bg = cv2.bitwise_and(origin, origin, mask=mask)
  cv2.imwrite(outputFilePath,img2_bg)
  print("===========Processing Is Done=============")
  print("Img Saved : "+outputFilePath)
  print("==========================================")


def start():
  print("PreProcessing Before Layer Extraction.")
  input_dir = "./input/"
  output_dir = "./input/"
  print("Enter the Input File Name : ", end='')
  InputFile = input()
  print("Enter the Output File Name : ", end='')
  OutputFile = "Non_bgr_"+InputFile.split('.')[0]+".png"
  inputFilePath = input_dir+InputFile
  outputFilePath = output_dir+OutputFile

  if inputFilePath is None or outputFilePath is None:
    print("Bad parameters. Please specify input file path and output file path")
    exit()
#modelType = "xception_model"
  modelType = "mobile_net_model"


  MODEL = DeepLabModel(modelType)
  print('model loaded successfully : ' + modelType)

  def run_visualization(filepath):
    """Inferences DeepLab model and visualizes result."""
    try:
      print("Trying to open : " + inputFilePath)
      # f = open(sys.argv[1])
      jpeg_str = open(filepath, "rb").read()
      orignal_im = Image.open(BytesIO(jpeg_str))
    except IOError:
      print('Cannot retrieve image. Please check file: ' + filepath)
      return

    print('running deeplab on image %s...' % filepath)
    resized_im, seg_map = MODEL.run(orignal_im)

    # vis_segmentation(resized_im, seg_map)
    drawSegment(resized_im, seg_map, inputFilePath, outputFilePath)

  run_visualization(inputFilePath)
  return outputFilePath
