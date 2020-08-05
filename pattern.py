import cv2
from PIL import Image

def set_alpha(R,G,B):
    val = (0.299*R) + (0.587*G) + (0.114*B)
    return int(val)

def avr_RGB(R,G,B):
    return int((R+G+B)/3)

def cal_pattern(source, pattern):
    p = source/255
    result = 0
    if(pattern>source):
        result= int(pattern*p)+int((pattern-source)/4)
    else:
        result = int(pattern*p)
    return result

def cal_contour_pattern(source, pattern):
    p = source/255
    result = 0
    if(pattern>source):
        try:
            result= int(pattern*p)-int((pattern-source)/4)
        except:
            result = 30*int((source+255)/255)
    else:
        result = int(pattern*p)
    return result

def canny_is_zero(a,b,c):
    if(a==0 and b == 0 and c==0):
        return True
    return False
def get_pattern(pattern_file):
    img = cv2.imread("./cloth.png")
    width = img.shape[1]
    height = img.shape[0]
    origin = cv2.imread(pattern_file)
    origin = cv2.resize(origin, (width,height))
    img2gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(img,70,240)
    contours, hierachy = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours)):
        cv2.drawContours(canny, [contours[i]], 0, (255, 255, 255), 1)
    ret, mask = cv2.threshold(img2gray, 127, 255, cv2.THRESH_BINARY)
    cv2.imwrite("cloth_mask.png", mask)
    mask_inverted = cv2.bitwise_not(mask)
    img1_bg = cv2.bitwise_and(origin, origin, mask=mask_inverted)
    cv2.imwrite("mask_inverted.png", mask_inverted)
    cv2.imwrite("cloth_pattern.png", img1_bg)
    cv2.imwrite("canny.png", canny)

    img = Image.open("cloth_pattern.png")  # 패턴그려진 옷마스킹
    origin = Image.open("./cloth.png") #옷열기
    test = Image.open("./canny.png")
    img = img.convert("RGBA")  # RGBA형식으로 변환
    origin = origin.convert("RGBA")
    test = test.convert("RGBA")
    pattern_datas = img.getdata()  # datas에 일차원 배열 형식으로 RGBA입력
    cloth_datas = origin.getdata()
    canny_datas = test.getdata()
    newData_pattern = []
    x = -1
    y = 0
    for i in range(len(pattern_datas)):
        if ((pattern_datas[i][0] == 0 and pattern_datas[i][1] == 0 and pattern_datas[i][2] == 0)):  # 해당 픽셀 색이 검정이거나, 턱선 위 일 경우 투명처리
            newData_pattern.append((255, 255, 255, 0))
        elif (not canny_is_zero(canny_datas[i][0],canny_datas[i][1],canny_datas[i][2])):
            R = cloth_datas[i][0]
            G = cloth_datas[i][1]
            B = cloth_datas[i][2]
            newData_pattern.append((cal_contour_pattern(R, pattern_datas[i][0]),cal_contour_pattern(G, pattern_datas[i][1]),cal_contour_pattern(B, pattern_datas[i][2]), 255))  # 해당 영역 추가
        else:  # 그렇지 않으면
            R = cloth_datas[i][0]
            G = cloth_datas[i][1]
            B = cloth_datas[i][2]
            newData_pattern.append((cal_pattern(R, pattern_datas[i][0]),cal_pattern(G, pattern_datas[i][1]),cal_pattern(B, pattern_datas[i][2]),255))  # 해당 영역 추가
    img.putdata(newData_pattern)  # 데이터 입력
    img.save("cloth_pattern.png")  # 이미지name으로 저장

