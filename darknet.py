from __future__ import print_function, division
from ctypes import *
import math
import random
import os
import cv2
import numpy as np
import time
import argparse

DISPLAY_COL = 640 # 640 1280
DISPLAY_ROW = 480 # 480 720

def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1

def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]

    
this_dir = os.path.dirname(__file__)
DATAPATH_DIR = this_dir + "/"
lib = CDLL(os.path.join(this_dir, "libdarknet.so"), RTLD_GLOBAL)

lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

# load_meta = lib.get_mapping
# lib.get_mapping.argtypes = [c_char_p, c_char_p]
# lib.get_mapping.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

#add numpy lize
ndarray_image = lib.ndarray_to_image
ndarray_image.argtypes = [POINTER(c_ubyte), POINTER(c_long), POINTER(c_long)]
ndarray_image.restype = IMAGE
#add numpy lize

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

#add numpy lize
def nparray_to_image(img):
    data = img.ctypes.data_as(POINTER(c_ubyte))
    image = ndarray_image(data, img.ctypes.shape, img.ctypes.strides)
    return image
#add numpy lize

def detect(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45):
    im = load_image(image, 0, 0)
    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
    num = pnum[0]
    if (nms): do_nms_obj(dets, num, meta.classes, nms);

    res = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    res = sorted(res, key=lambda x: -x[1])
    free_image(im)
    free_detections(dets, num)
    return res

def detect_np(net, meta, np_img, thresh=.5, hier_thresh=.5, nms=.45):
    '''
    detect the image by numpy array image format
    '''
    im = nparray_to_image(np_img)
    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
    num = pnum[0]
    if (nms): do_nms_obj(dets, num, meta.classes, nms);

    res = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                res.append((meta.names[i], dets[j].prob[i], (int(b.x-0.5*b.w), int(b.y-0.5*b.h), int(b.x+0.5*b.w), int(b.y+0.5*b.h))))
    res = sorted(res, key=lambda x: -x[1])
    free_image(im)
    free_detections(dets, num)
    return res


class yolo_dn(object):
    def __init__ (self, height, width, batchsize, threshold, cfg_path, weight_path, meta_path):
        self.height = height
        self.width = width
        self.batchsize = batchsize
        self.threshold = threshold

        self.cfg_file = os.path.join(this_dir, cfg_path)
        self.meta_file = os.path.join(this_dir, meta_path)
        self.weights_file = os.path.join(this_dir, weight_path)
        
        print("load yolo model ...")
        self.net = load_net(self.cfg_file, self.weights_file, 0)
        self.meta = load_meta(self.meta_file)#, DATAPATH_DIR)
        print("load yolo model done ...")
    def __call__(self, bgr_image): # detect
        '''
        input: numpy array with uint8 datatype. 
            hwc order numpy array image with bgr channel order.
        return: List of results
            list of tuple: ( label, prob, (x1, y1, x2, y2))
        '''
        return detect_np(self.net, self.meta, bgr_image, thresh=self.threshold)
    def __del__(self):
        print ("destory detector")

def create_yolo_c(height, width, batchsize=1, threshold=0.25,
                cfg_path="/home/jacob/yolo_darknet/cfg/yolov3.cfg", weight_path="/home/jacob/yolo_darknet/yolov3.weights", meta_path="/home/jacob/yolo_darknet/cfg/coco.data"):
    print("threshold: ",threshold)
    yoloDarknet_detector = yolo_dn(height, width, batchsize, threshold, cfg_path, weight_path, meta_path)
    def detector(bgr_image):
        return yoloDarknet_detector(bgr_image)
    return detector


def draw_boxes(image, detections, color=(0,255,0), draw_conf=False):
    for det in detections:
        if (draw_conf is True) and (det[0] is not None):
            label = "<{:}>: {:.2f}%".format(det[0].decode('utf-8'), det[1]*100)
            cv2.putText(image, label, (det[2][0], det[2][1]-5), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), lineType=cv2.LINE_AA)
            cv2.rectangle(image, 
            (det[2][0], det[2][1]), (det[2][2], det[2][3]),
            color, 1, cv2.LINE_AA)

def draw_str(dst, target, s):
    x, y = target
    cv2.putText(dst, s, (x+1, y+1), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness = 2, lineType=cv2.LINE_AA)
    cv2.putText(dst, s, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)


def clock():
    return cv2.getTickCount() / cv2.getTickFrequency()

class StatValue:
    def __init__(self, smooth_coef = 0.5):
        self.value = None
        self.smooth_coef = smooth_coef
    def update(self, v):
        if self.value is None:
            self.value = v
        else:
            c = self.smooth_coef
            self.value = c * self.value + (1.0-c) * v

def checkpath(fullpath):
    path = os.path.dirname(fullpath)
    if not os.path.exists(path):
        os.makedirs(path)



def run_video(args):

    maxFPS = 25
    time_frame = 1000/ int(maxFPS)

    row = 720  # 1080 480 720
    col = 1280 # 1920 640 1280

    DisplayByStream = False

    # files for yolo detection model
    cfg_file = args.cfg
    meta_file = args.meta
    weights_file = args.weights

    # threshold for classification and detection
    threshold_det = args.threshold_det 
    threshold_cls = args.threshold_cls

    # input image and output path
    video_path = args.video
    output_path = args.out

    save_video = False
    if output_path:
        checkpath(output_path)
        save_video = True
    
    # load the video
    cap = cv2.VideoCapture("2.mp4", cv2.CAP_FFMPEG)

    # Check if camera opened successfully
    if (cap.isOpened()== False):
        print("Error opening video stream or file")
        raise ValueError("video is not opened.")

    if save_video == True:
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('output.mp4',fourcc, 20.0, (col,row))

    # load yolo model
    print("load yolo model ...")
    net = load_net(cfg_file, weights_file, 0)
    meta = load_meta(meta_file)
    print("load yolo model done ...")

    # load classifier
    # classifer = deep_disk.create_disk_classifier(threshold=threshold_cls)

    latency = StatValue()
    currentFrame = time.time()
    lastFrame = time.time()
    delta = 0
    while(cap.isOpened()):
        ret, img = cap.read()
        if ret==False:
            print("Finished.")
            break
        currentFrame = time.time()
        delta = (currentFrame - lastFrame) * 1000
        lastFrame = currentFrame

        t = clock()
        img = cv2.resize(img, (col, row))
        det = detect_np(net, meta, img, thresh=threshold_det)
        draw_boxes(img, det, (0,255,0), draw_conf=True)

        latency.update(clock() - t)
        draw_str(img, (20, 40), "latency        :  %.1f ms" % (latency.value * 1000))
        draw_str(img, (20, 60), "times          :  %.1f ms" % (delta))


        if DisplayByStream:
            img = cv2.resize(img, (DISPLAY_COL, DISPLAY_ROW)) # remeber to rezie the image to specified size before boardcast
            #client.send(img.tostring())
        else:
            cv2.imshow('Display', img)

        if save_video == True:
            # write the frame
            out.write(img)

        if delta < time_frame:
            time.sleep((time_frame - delta)/1000)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    
    if save_video == True:
        out.release()
    cap.release()
    cv2.destroyAllWindows()

    raise ValueError("streaming is stop!.")

    print("Streaming done")



def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Inference for TNB")
    parser.add_argument(
        "--cfg", help=" path to the darknet cfg file for deployment model ",
        default="/home/jacob/yolo_darknet/cfg/yolov3.cfg".encode('utf-8'), type=str)
    parser.add_argument(
        "--meta", help=" path to the darknet meta file for deployment model ",
        default="/home/jacob/yolo_darknet/cfg/coco.data".encode('utf-8'), type=str)
    parser.add_argument(
        "--weights", help=" path to the darknet weights file for deployment model ",
        default="/home/jacob/yolo_darknet/yolov3.weights".encode('utf-8'), type=str)
    parser.add_argument(
        "--threshold_det", help=" threshold for detection model ",
        default=0.7, type=float)
    parser.add_argument(
        "--threshold_cls", help=" threshold for classifcation model ",
        default=0.9, type=float)
    parser.add_argument(
        "--img", help=" path to the input image ",
        default=None, type=str)
    parser.add_argument(
        "--video", help=" path to the input video ",
        default="2.mp4", type=str)
    parser.add_argument(
        "--out", help=" output path for the output ",
        default=None, type=str)
    #parser.add_argument('--rename', dest='rename', help="rename the file's name to avoid same-name file.",action='store_true')
    return parser.parse_args()

if __name__ == "__main__":


    args = parse_args()
    run_video(args)




        

