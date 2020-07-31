# MakeUpProject
게임 리소스[아바타 및 가상 인물] 생성 AI 프로젝트

## Description
- 게임 리소스 생성을 위한 AI기반 가상 얼굴 생성 및 Face Part Layer 추출

## Env
- Based On Unity Project
- Python 3.6
- Open CV 4.3.0
- Tensorflow 2.2.0
- numpy 1.17.4
- Illustration : CartoonGAN
- dlib 19.20.0
- matplotlib 3.2.2
- Pillow 7.2.0


## Progress
- [X] Unity 상에서의 FaceRecognition 구현, Face Part별 화장 기능 구현 예정 [20.07.10]
- [X] Face LandMark 부위별 추출 완료 [20.07.13]
- [X] 프로젝트 내용 변경 : 게임 리소스 활용을 위한 Face Layer 추출 및 Face Part Layer 추출[20.07.20]
- [X] Face Layer, Cheek Layer, eyebrow Layer추출 구현[20.07.21]
- [X] 섀도우, 눈 레이어 추가, 눈썹 레이어 수정[20.07.21]
- [X] CartoonGAN 적용, 실사 사진 일러스트화적용 [20.07.23]
- [X] CartoonGAN , Layer Extraction 모듈 통합 [20.07.27]
- [X] Clothes Layer Extraction구현(상의) [20.07.27]
- [ ] Clothes Layer Extraction구현(하의) 
- [ ] Body Layer Extraction
- [ ] Convert 2D Pattern to 3D style

#### CartoonGAN processing 전/후
 <img src= "./Resources/21.jpg" width="212px"> <img src= "./Resources/20.jpg" width="212px">

#### FaceLayer Extraction
 <img src= "./Resources/5.png" width="400px"> <img src= "./Resources/6.png" width="212px">

#### Cheek Layer Extraction
 <img src= "./Resources/cheek_colored2.0.png" width="530px"> 

#### EyeBrow Layer Extraction
  <img src= "./Resources/eye_layer.PNG">

### EyeShadow Layer Extraction
<img src= "./Resources/eyeshadow_layer.PNG">

#### Clothes Layer Extraction [h5 Model from Anish Josh]
  <img src= "./Resources/clothes_layer.png" width="550px">
  <img src= "./Resources/out.png" width="550px">
  
```
- topwears.h5 model file download
https://drive.google.com/file/d/14vTYmsHjUYv3VPo1Byrecs3NQuvJo89t/view
```

## Revision History
- Initialize Project [20.07.11]
- Modified : Face Part LandMark Extraction  [20.07.13]
- Added : Face Layer Extraction Module [20.07.20]
- Added : Eyebrow, Cheek Layer Extraction Module [20.07.21]
- Modified : Shadow and Eye Layer, Modification Eyebrow Layer [20.07.22]
- Added : CartoonGAN Module [20.07.23]
- Integrated : CartoonGAN & Face Layer Extraction Module [20.07.27]
- Added : Clothes Layer Extraction [20.07.27]
- Modified : Face Skin, Lip Layer Extraction Accuracy Improvement [20.07.31]
- Modified : Eye shadow layer can be extracted from the whole body picture [20.07.31]

## Lisence
MIT License

Copyright (c) [2020] [LeeJaeBeen]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.