import numpy as np
import cv2
import os
from utils import linear_mapping, pre_process, random_warp

"""
This module implements the basic correlation filter based tracking algorithm -- MOSSE
基于滤波的跟踪就是用在模板图片上训练好的滤波器去对目标物体的外表建模。目标最初是基于以第一帧中的目标
为中心的一个小跟踪窗口来选择的。从这点上来说，跟踪器和滤波器训练是一起进行的。通过在下一帧图片的搜索
窗口中去进行滤波来跟踪目标。滤波之后产生的最大值的地方就是目标的新位置。根据得到的新位置完成在线更新。

Correlation Filter应用于tracking方面最朴素的想法就是：相关是衡量两个信号相似值的度量，如果两个信号越相似，那么其相关值就越高，
而在tracking的应用里，就是需要设计一个滤波模板，使得当它作用在跟踪目标上时，得到的响应最大
"""


class Mosse:
    def __init__(self, args, img_path):
        # get arguments..
        self.args = args
        self.img_path = img_path
        # get the img lists...
        self.frame_lists = self._get_img_lists(self.img_path)
        self.frame_lists.sort()

    # start to do the object tracking...
    def start_tracking(self):
        # get the image of the first frame... (read as gray scale image...)
        # 读取到初始的第一帧图像，然后将图像由 BGR 转变为 GRAY 图像
        init_img = cv2.imread(self.frame_lists[0])
        init_frame = cv2.cvtColor(init_img, cv2.COLOR_BGR2GRAY)
        init_frame = init_frame.astype(np.float32)

        # get the init ground truth.. [x, y, width, height]
        # 这里通过手工框出想要选择的目标区域
        init_gt = cv2.selectROI('demo', init_img, False, False)
        init_gt = np.array(init_gt).astype(np.int64)

        # start to draw the gaussian response...
        # 得到高斯响应图（输入原始图像以及目标区域的位置 [x, y, width, height]）返回高斯函数矩阵，在选定的目标框的中心，其值最大
        response_map = self._get_gauss_response(init_frame, init_gt)

        # start to create the training set ...
        # get the goal...
        # 抽取框选区域的图像及其高斯响应
        g = response_map[init_gt[1]:init_gt[1] + init_gt[3], init_gt[0]:init_gt[0] + init_gt[2]]
        fi = init_frame[init_gt[1]:init_gt[1] + init_gt[3], init_gt[0]:init_gt[0] + init_gt[2]]
        # 对目标区域的高斯响应图做快速傅立叶变换
        G = np.fft.fft2(g)

        # 做滤波器的预训练
        # start to do the pre-training...
        Ai, Bi = self._pre_training(fi, G)

        # start the tracking...
        for idx in range(len(self.frame_lists)):

            current_frame = cv2.imread(self.frame_lists[idx])
            frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            frame_gray = frame_gray.astype(np.float32)

            if idx == 0:
                Ai = self.args.lr * Ai
                Bi = self.args.lr * Bi
                pos = init_gt.copy()
                clip_pos = np.array([pos[0], pos[1], pos[0] + pos[2], pos[1] + pos[3]]).astype(np.int64)
            else:
                Hi = Ai / Bi
                fi = frame_gray[clip_pos[1]:clip_pos[3], clip_pos[0]:clip_pos[2]]
                fi = pre_process(cv2.resize(fi, (init_gt[2], init_gt[3])))

                Gi = Hi * np.fft.fft2(fi)
                # 对于频域下的 Gi 进行逆傅立叶变换得到实际的 gi
                gi = linear_mapping(np.fft.ifft2(Gi))

                # find the max pos...
                max_value = np.max(gi)
                # 获取到 gi 中最大值的坐标，这个位置就是当前帧中目标的坐标，只不过这个坐标是相对于 gi，也就是目标区域而言的
                max_pos = np.where(gi == max_value)
                # gi.shape[0] / 2 就是上一个目标的 y 坐标，也是相对于 gi 这个区域而言，相减得到的 dy 就是当前目标与上一个目标在 y 方向的偏移量
                dy = int(np.mean(max_pos[0]) - gi.shape[0] / 2)
                # gi.shape[1] / 2 就是上一个目标的 x 坐标，也是相对于 gi 这个区域而言，相减得到的 dx 就是当前目标与上一个目标在 x 方向的偏移量
                dx = int(np.mean(max_pos[1]) - gi.shape[1] / 2)

                # update the position...
                pos[0] = pos[0] + dx
                pos[1] = pos[1] + dy

                # trying to get the clipped position [xmin, ymin, xmax, ymax]
                # clip_pos 表示的是在这一帧中，目标区域的新位置 [leftX, topY, rightX, bottomY]
                clip_pos[0] = np.clip(pos[0], 0, current_frame.shape[1])
                clip_pos[1] = np.clip(pos[1], 0, current_frame.shape[0])
                clip_pos[2] = np.clip(pos[0] + pos[2], 0, current_frame.shape[1])
                clip_pos[3] = np.clip(pos[1] + pos[3], 0, current_frame.shape[0])
                clip_pos = clip_pos.astype(np.int64)

                # get the current fi..
                fi = frame_gray[clip_pos[1]:clip_pos[3], clip_pos[0]:clip_pos[2]]
                fi = pre_process(cv2.resize(fi, (init_gt[2], init_gt[3])))

                # online update...
                # 在线更新 Ai, Bi
                Ai = self.args.lr * (G * np.conjugate(np.fft.fft2(fi))) + (1 - self.args.lr) * Ai
                Bi = self.args.lr * (np.fft.fft2(fi) * np.conjugate(np.fft.fft2(fi))) + (1 - self.args.lr) * Bi

            # visualize the tracking process...
            cv2.rectangle(current_frame, (pos[0], pos[1]), (pos[0] + pos[2], pos[1] + pos[3]), (255, 0, 0), 2)
            cv2.imshow('demo', current_frame)
            cv2.waitKey(100)

            # if record... save the frames..
            if self.args.record:
                frame_path = 'record_frames/' + self.img_path.split('/')[1] + '/'
                if not os.path.exists(frame_path):
                    os.mkdir(frame_path)
                cv2.imwrite(frame_path + str(idx).zfill(5) + '.png', current_frame)

    # pre train the filter on the first frame...
    def _pre_training(self, init_frame, G):
        height, width = G.shape
        fi = cv2.resize(init_frame, (width, height))
        # pre-process img..
        fi = pre_process(fi)

        # np.fft.fft2 表示求 fi 的傅立叶变换
        # np.conjugate 表示求矩阵的共轭
        # 比如 g = np.matrix('[1+2j, 2+3j; 3-2j, 1-4j]')
        # g.conjugate 为 matrix([[1-2j, 2-3j],[3+2j, 1+4j]])
        Ai = G * np.conjugate(np.fft.fft2(fi))
        Bi = np.fft.fft2(init_frame) * np.conjugate(np.fft.fft2(init_frame))

        # 对 fi 进行多次刚性形变，增强检测的鲁棒性，计算出 Ai 和 Bi 的初始值
        for _ in range(self.args.num_pretrain):
            if self.args.rotate:
                fi = pre_process(random_warp(init_frame))
            else:
                fi = pre_process(init_frame)
            Ai = Ai + G * np.conjugate(np.fft.fft2(fi))
            Bi = Bi + np.fft.fft2(fi) * np.conjugate(np.fft.fft2(fi))

        return Ai, Bi

    # get the ground-truth gaussian response...
    def _get_gauss_response(self, img, gt):
        # get the shape of the image..
        height, width = img.shape
        # get the mesh grid...
        xx, yy = np.meshgrid(np.arange(width), np.arange(height))

        # get the center of the object...
        # 得到选定的目标区域的中心点坐标
        center_x = gt[0] + 0.5 * gt[2]
        center_y = gt[1] + 0.5 * gt[3]

        # cal the distance...
        # 创建一个以选定的目标中点为中心，且符合二维高斯分布的响应矩阵，矩阵大小等于原图像 img 的大小
        exponent = (np.square(xx - center_x) + np.square(yy - center_y)) / (2 * self.args.sigma)
        # get the response map...
        response = np.exp(-exponent)

        # normalize...
        # 对响应矩阵进行归一化处理: (x - min) / (max - min)
        response = linear_mapping(response)
        return response

    # it will extract the image list 
    @staticmethod
    def _get_img_lists(img_path):
        frame_list = []
        for frame in os.listdir(img_path):
            if os.path.splitext(frame)[1] == '.jpg':
                frame_list.append(os.path.join(img_path, frame))
        return frame_list

    # it will get the first ground truth of the video..
    @staticmethod
    def _get_init_ground_truth(img_path):
        gt_path = os.path.join(img_path, 'groundtruth.txt')
        with open(gt_path, 'r') as f:
            # just read the first frame...
            line = f.readline()
            gt_pos = line.split(',')

        return [float(element) for element in gt_pos]
