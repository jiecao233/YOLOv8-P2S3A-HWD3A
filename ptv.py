import cv2 as cv
import os

def image_to_video():
    file_path = 'K:/data_ball/train/images/'  # 图片目录
    output = 'K:/data_ball/train/images/heatmap.mp4'  # 生成视频路径
    img_list = os.listdir(file_path)  # 生成图片目录下以图片名字为内容的列表
    height = 720
    weight = 1280
    fps = 60
    # fourcc = cv.VideoWriter_fourcc('M', 'J', 'P', 'G') 用于avi格式的生成
    fourcc = cv.VideoWriter_fourcc(*'mp4v')  # 用于mp4格式的生成
    videowriter = cv.VideoWriter(output, fourcc, fps, (weight, height))  # 创建一个写入视频对象
    for img in img_list:
        path = file_path + img
        # print(path)
        frame = cv.imread(path)
        videowriter.write(frame)

    videowriter.release()

image_to_video()