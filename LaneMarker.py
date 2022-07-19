import os
import random

import cv2
import numpy as np
from scipy.interpolate import UnivariateSpline
  

class LaneMarker:

    def __init__(self, data_dir_path, EXEC_ROOT_PATH):
        # all dir path ends with slash
        self.data_dir_path = data_dir_path + "/" # has to be abs path
        self.raw_image_dir_path = self.data_dir_path + "raw_image/"
        self.binary_dir_path = self.data_dir_path + "gt_binary_image/"
        self.instance_dir_path = self.data_dir_path + "gt_instance_image/"
        self.train_file_path = self.data_dir_path + "train.txt"
        self.val_file_path = self.data_dir_path + "val.txt"

        #
        self.EXEC_ROOT_PATH = EXEC_ROOT_PATH + "/" 
        self.EXEC_IMG_DIR_PATH = self.EXEC_ROOT_PATH + "raw_image/"
        self.EXEC_BIN_DIR_PATH = self.EXEC_ROOT_PATH + "gt_binary_image/"
        self.EXEC_INS_DIR_PATH = self.EXEC_ROOT_PATH + "gt_instance_image/"

        #
        self.cur_img = None
        self.cur_img_NAME = None
        self.single_lane_pts = []
        self.full_lane_pts = []

        #
        self.color_set = [
            (40,40,40),
            (80,80,80),
            (120,120,120),
            (160,160,160)
        ]
    

    def parse_img(self, img_path, params):

        print("Parsing image " + self.cur_img_NAME + " ...")

        self.cur_img = cv2.imread(img_path, 1)
        cv2.imshow(self.cur_img_NAME, self.cur_img)

        cv2.setMouseCallback(self.cur_img_NAME, self.click_event, params)
    
        # wait for a key to be pressed to exit
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        

    def click_event(self, event, x, y, flags, params):
        # checking for left mouse clicks
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(self.cur_img, (x,y), 5, (0, 0, 255), -1)
            cv2.imshow(self.cur_img_NAME, self.cur_img)
            self.add_lane(x, y, params[0], params[1], params[2])


    def add_profile(self, file_type):
        if file_type == 't':
            with open(self.train_file_path, 'a') as f:
                f.write(self.EXEC_IMG_DIR_PATH+self.cur_img_NAME+" "+self.EXEC_BIN_DIR_PATH+self.cur_img_NAME+" "+self.EXEC_INS_DIR_PATH+self.cur_img_NAME+" \n")
        elif file_type == 'v':
            with open(self.val_file_path, 'a') as f:
                f.write(self.EXEC_IMG_DIR_PATH+self.cur_img_NAME+" "+self.EXEC_BIN_DIR_PATH+self.cur_img_NAME+" "+self.EXEC_INS_DIR_PATH+self.cur_img_NAME+" \n")


    def add_lane(self, x, y, pts_num, lane_num, file_type):
        self.single_lane_pts.append([x,y])

        print("Adding point (" + str(x) + "," + str(y) + "), current lane: " + str(self.single_lane_pts))
        print("Number of points left in lane: " + str(pts_num - len(self.single_lane_pts)))

        if len(self.single_lane_pts) == pts_num:
            self.full_lane_pts.append(np.array(self.single_lane_pts))
            # single lane reset
            self.single_lane_pts = []

        print("Number of lane left in image: " + str(lane_num - len(self.full_lane_pts)))
        
        if len(self.full_lane_pts) == lane_num:
            self.full_lane_pts = np.array(self.full_lane_pts)
            print("Full lanes: " + str(self.full_lane_pts))
            self.lane_processing()
            self.add_profile(file_type)
            # full lane pts reset
            self.full_lane_pts = []


    def lane_processing(self):
        binary_canvas = np.zeros((720, 1280, 3), np.uint8)
        instance_canvas = np.zeros((720, 1280, 3), np.uint8)

        if len(self.full_lane_pts[0]) <= 2:
            print("Generating segmentation images...")
            for i in range(len(self.full_lane_pts)):
                lane = self.full_lane_pts[i]
                binary_canvas = cv2.line(binary_canvas, tuple(lane[0]), tuple(lane[1]), (255, 255, 255), thickness=5)
                instance_canvas = cv2.line(instance_canvas, tuple(lane[0]), tuple(lane[1]), self.color_set[i], thickness=5)

            img_new_NAME = 'd'+self.cur_img_NAME[1:]
            os.rename(self.raw_image_dir_path+self.cur_img_NAME, self.raw_image_dir_path+img_new_NAME)
            self.cur_img_NAME = img_new_NAME

            cv2.imshow('b', binary_canvas)
            cv2.imshow('i', instance_canvas)
            print("Writing segmentation images...")
            cv2.imwrite(self.binary_dir_path+img_new_NAME, binary_canvas)
            cv2.imwrite(self.instance_dir_path+img_new_NAME, instance_canvas)   
            print("Segmentation images writing complete.\n")
        else:
            print("Fitting splines...")
            splines = []
            for lane in self.full_lane_pts:
                splines.append(UnivariateSpline(lane[:,1],lane[:,0]))
            
            print("Generating segmentation images...")
            for i in range(len(splines)):
                dy = np.linspace(self.full_lane_pts[i,-1,1], self.full_lane_pts[i,0,1], 100)
                sp = splines[i]
                sp.set_smoothing_factor(5.0)
                curve_segs = np.dstack((sp(dy), dy)).reshape((len(dy),2)).astype(int)
                for j in range(len(dy)):
                    if j == 0: continue
                    binary_canvas = cv2.line(binary_canvas, tuple(curve_segs[j-1]), tuple(curve_segs[j]), (255, 255, 255), thickness=5)
                    instance_canvas = cv2.line(instance_canvas, tuple(curve_segs[j-1]), tuple(curve_segs[j]), self.color_set[i], thickness=5)

            img_new_NAME = 'd'+self.cur_img_NAME[1:]
            os.rename(self.raw_image_dir_path+self.cur_img_NAME, self.raw_image_dir_path+img_new_NAME)
            self.cur_img_NAME = img_new_NAME

            cv2.imshow('b', binary_canvas)
            cv2.imshow('i', instance_canvas)
            print("Writing segmentation images...")
            cv2.imwrite(self.binary_dir_path+img_new_NAME, binary_canvas)
            cv2.imwrite(self.instance_dir_path+img_new_NAME, instance_canvas)   
            print("Segmentation images writing complete.\n")


    def run_marker(self):
        images = os.listdir(self.raw_image_dir_path)
        img_s = [] # straight lane
        img_c = [] # curve lane

        for img in images:
            if img[0] == 'i':
                if img[1] == 's': img_s.append(img)
                elif img[1] == 'c': img_c.append(img)

        random.shuffle(img_s)
        random.shuffle(img_c)

        for img in img_s:
            self.cur_img_NAME = img
            if int(self.s_cur) < int(self.st_count):
                params = [2, int(img[5]), 't']
            else: 
                params = [2, int(img[5]), 'v']
            self.parse_img(self.raw_image_dir_path+img, params)
            self.s_cur = str(int(self.s_cur)+1)
            self.write_count()
        
        for img in img_c:
            self.cur_img_NAME = img
            if int(self.c_cur) < int(self.ct_count):
                params = [12, int(img[5]), 't']
            else: 
                params = [12, int(img[5]), 'v']
            self.parse_img(self.raw_image_dir_path+img, params)
            self.c_cur = str(int(self.c_cur)+1)
            self.write_count()
    

    def init_count(self):
        with open(self.data_dir_path+"tv_count.txt", 'r') as f:
            counts = f.read().split(';')
            self.st_count, self.s_cur = tuple(counts[0].split(','))
            self.ct_count, self.c_cur = tuple(counts[1].split(','))
            print(int(self.ct_count))
    
    def write_count(self):
        with open(self.data_dir_path+"tv_count.txt", 'w') as f:
            f.write(self.st_count+','+self.s_cur+';'+self.ct_count+','+self.c_cur)