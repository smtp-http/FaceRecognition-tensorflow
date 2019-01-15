import cv2
import dlib
import os
import sys
#import random
import time
from multiprocessing import Process, JoinableQueue, Queue
from random import random
import time

#tasks_queue = JoinableQueue()
results_queue = Queue()



name = "abc"#input('please input your name:')
window_name = "face picture entry"
output_dir = './' + name + '_faces'
size = 64

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 改变图片的亮度与对比度
def relight(img, light=1, bias=0):
    w = img.shape[1]
    h = img.shape[0]
    #image = []
    for i in range(0,w):
        for j in range(0,h):
            for c in range(3):
                tmp = int(img[j,i,c]*light + bias)
                if tmp > 255:
                    tmp = 255
                elif tmp < 0:
                    tmp = 0
                img[j,i,c] = tmp
    return img

#使用dlib自带的frontal_face_detector作为我们的特征提取器
detector = dlib.get_frontal_face_detector()

#videoName = './' + name + '.avi'
videoName = "rtsp://admin:ABC_123456@172.17.208.150:554/Streaming/Channels/101?transportmode=unicast"
# 打开摄像头 参数为输入流，可以为摄像头或视频文件


def CatchUsbVideo(out_queue):
    process_flag = False
    cv2.namedWindow(window_name)
 
    #视频来源，可以来自一段已存好的视频，也可以直接来自USB摄像头
    cap = cv2.VideoCapture(videoName)      
    n = 1
    while cap.isOpened():
        ok, frame = cap.read() #读取一帧数据
        if not ok:            
            break                    
 
        #显示图像并等待10毫秒按键输入，输入‘q’退出程序
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL) #cv2.WND_PROP_FULLSCREEN)
        #cv2.moveWindow(window_name, screen.x - 1, screen.y - 1)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN,
                          cv2.WINDOW_FULLSCREEN)
        n = n + 1
        if n == 20:
            if process_flag == True:
                out_queue.put(frame)
            n = 1
            
        else:
            frameresize = cv2.resize(frame,(1280,800))
            cv2.imshow(window_name, frameresize)

        c = cv2.waitKey(10)
        if c & 0xFF == ord('q'):
            break        
        if c & 0xFF == ord('d'):
            process_flag = True
 
    #释放摄像头并销毁所有窗口
    cap.release()
    cv2.destroyAllWindows() 


def FaceDectection(in_queue):
    while 1:
        img = in_queue.get()
        if img is None:
            break
        else:
            print("====== recv pic")


#if __name__ == '__main__':
#    CatchUsbVideo()
processes = []

p = Process(target=CatchUsbVideo, args=(results_queue,))
p.start()
processes.append(p)

p = Process(target=FaceDectection, args=(results_queue,))
p.start()
processes.append(p)


for p in processes:
    p.join()

while 1:
    time.sleep(1)

"""
index = 2499
while True:
    if (index <= 10000):
        print('Being processed picture %s' % index)
        # 从摄像头读取照片
        success, img = camera.read()
        # 转为灰度图片
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 使用detector进行人脸检测
        dets = detector(gray_img, 1)

        for i, d in enumerate(dets):
            x1 = d.top() if d.top() > 0 else 0
            y1 = d.bottom() if d.bottom() > 0 else 0
            x2 = d.left() if d.left() > 0 else 0
            y2 = d.right() if d.right() > 0 else 0

            face = img[x1:y1,x2:y2]
            # 调整图片的对比度与亮度， 对比度与亮度值都取随机数，这样能增加样本的多样性
            face = relight(face, random.uniform(0.5, 1.5), random.randint(-50, 50))

            face = cv2.resize(face, (size,size))

            cv2.imshow('image', face)

            cv2.imwrite(output_dir+'/'+str(index)+'.jpg', face)

            index += 1
        key = cv2.waitKey(30) & 0xff
        if key == 27:
            break
    else:
        print('Finished!')
        break
"""