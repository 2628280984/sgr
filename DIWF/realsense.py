import argparse  # python的命令行解析的标准模块  可以让我们直接在命令行中就可以向程序中传入参数并让程序运行
import os
import shutil
import time
from pathlib import Path  # Path将str转换为Path对象 使字符串路径易于操作的模块

import cv2
import torch
import torch.backends.cudnn as cudnn  # cuda模块
from numpy import random
import numpy as np
import pyrealsense2 as rs  # 导入realsense的sdk模块

from models.experimental import attempt_load
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, plot_one_box, strip_optimizer, set_logging)
from utils.torch_utils import select_device, load_classifier, time_synchronized
from utils.datasets import letterbox


def detect(save_img=False):
    # 加载参数
    out, source, weights, view_img, save_txt, imgsz = \
        opt.save_dir, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source == '0' or source.startswith(('rtsp://', 'rtmp://', 'http://')) or source.endswith('.txt')

    # 初始化
    set_logging()  # 生成日志
    device = select_device(opt.device)  # 获取当前主机可用的设备
    if os.path.exists(out):  # output dir
        shutil.rmtree(out)  # delete dir
    os.makedirs(out)  # make new dir
    # 如果设配是GPU 就使用half(float16)  包括模型半精度和输入图片半精度
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # 载入模型和模型参数并调整模型
    model = attempt_load(weights, map_location=device)  # 加载Float32模型
    imgsz = check_img_size(imgsz, s=model.stride.max())  # 确保输入图片的尺寸imgsz能整除stride=32 如果不能则调整为能被整除并返回
    if half:  # 是否将模型从float32 -> float16  加速推理
        model.half()  # to FP16

    # 加载推理数据
    vid_path, vid_writer = None, None
    # 采用webcam数据源
    view_img = True
    cudnn.benchmark = True  # 加快常量图像大小推断
    # dataset = LoadStreams(source, img_size=imgsz)  #load 文件夹中视频流

    # 获取每个类别的名字和随机生成类别颜色
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # 正式推理
    if device.type != 'cpu':
        model(torch.zeros(1, 6, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()

    # 实例化realsense模块
    # https://www.rs-online.com/designspark/intelpython2-nvidia-jetson-nanorealsense-d435-cn
    pipeline = rs.pipeline()
    # 创建 config 对象：
    config = rs.config()
    # 声明RGB和深度视频流
    # config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # 启动数据流
    pipeline.start(config)
    align_to_color = rs.align(rs.stream.color)  # 对齐rgb和深度图
    while True:
        start = time.time()
        frames = pipeline.wait_for_frames()
        aligned_frames = align_to_color.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not aligned_depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # combined_image = np.concatenate((color_image, depth_image), axis=2)

        # 对RGB的img进行处理，送入预测模型
        sources = [source]  # 数据源
        path = sources  # path: 图片/视频的路径

        imgs = [None]
        imgs1 = [None]

        imgs[0] = color_image
        imgs1[0] = depth_image

        im0s = imgs.copy()  # img0s: 原尺寸的图片
        im0s1 = imgs1.copy()  # img0s: 原尺寸的图片

        img = [letterbox(x, new_shape=imgsz)[0] for x in im0s]  # img: 进行resize + pad之后的图片
        img1 = [letterbox(x, new_shape=imgsz)[0] for x in im0s1]

        img = np.stack(img, 0)  # 沿着0dim进行堆叠
        img1 = np.stack(img1, 0)  # 沿着0dim进行堆叠


        img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)  # 从HWC变为CHW
        img1 = img1[:, :, :, ::-1].transpose(0, 3, 1, 2)  # 从HWC变为CHW

        # img = np.ascontiguousarray(img, dtype=np.float16 if half else np.float32)
        img = np.ascontiguousarray(img)
        img1 = np.ascontiguousarray(img1)


        # 处理每一张图片的数据格式
        img = torch.from_numpy(img).to(device)
        img1 = torch.from_numpy(img1).to(device)

        img = torch.cat((img, img1), 1)

        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        # 如果图片是3维(RGB) 就在前面添加一个维度1当中batch_size=1
        # 因为输入网络的图片需要是4为的 [batch_size, channel, w, h]
        if img.ndimension() == 3:
            img = img.unsqueeze(0)  # 在dim0位置添加维度1，[channel, w, h] -> [batch_size, channel, w, h]

        t1 = time_synchronized()  # 精确计算当前时间  并返回当前时间
        # 对每张图片/视频进行前向推理
        pred = model(img, augment=opt.augment)[0]

        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # 后续保存或者打印预测信息
        for i, det in enumerate(pred):  # detections per image
            p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()

            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    # s += '%g %ss, ' % (n, names[int(c)])  # add to string
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # 归一化为 xywh
                    line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                    label = f'{names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)  # 将结果框打印回原图

            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results
            if view_img:
                cv2.imshow('camera', im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration
    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--weights', nargs='+', type=str, default='yolov5m.pt', help='model.pt path(s)')
    parser.add_argument('--weights', nargs='+', type=str,default='best.pt',help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-dir', type=str, default='inference/output', help='directory to save results')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():  # 一个上下文管理器，被该语句wrap起来的部分将不会track梯度
        detect()

