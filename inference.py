import argparse
import os
import io
import time

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image

from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import label_map_util

if tf.__version__ < '1.4.0':
    raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')

NUM_CLASSES = 90


def parse_args(check=True):
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--input_dir', type=str, required=True)
    FLAGS, unparsed = parser.parse_known_args()
    return FLAGS, unparsed

def load_label_to_list(label_path):
    f = open(label_path, encoding="utf-8")
    categories = []
    category_line = {}
    line = f.readline()
    while line:
        line = line.strip('\n')
        line_info = line.split(':')
        line_id = int(line_info[0])
        line_name = line_info[1]
        category_line = {'id': line_id, 'name': line_name}
        categories.append(category_line)
        line = f.readline()
    f.close()
    return categories

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

def mainfunction(test_img_path):
    # FLAGS, unparsed = parse_args()
    # PATH_TO_CKPT = os.path.join('output_dir', 'exported_graphs/frozen_inference_graph.pb')
    # PATH_TO_LABELS = os.path.join('input_dir', 'labels_items.txt')

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile('input_dir/frozen_inference_graph_4.pb', 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    #读取pytxt为json
    #label_map = label_map_util.load_labelmap('input_dir/mscoco_label_map.pbtxt')
    #每一类作为字典放到list
    #categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    categories = load_label_to_list('input_dir/labels.txt')
    category_index = label_map_util.create_category_index(categories)

    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            image = Image.open(test_img_path)
            image_np = load_image_into_numpy_array(image)
            image_np_expanded = np.expand_dims(image_np, axis=0)
            point1=time.time()
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            point2 = time.time()
            print(point2 - point1)
            #######################################################################
            # 去掉size为1的维度
            boxes_squeeze = np.squeeze(boxes)
            scores_squeeze = np.squeeze(scores)
            classes_squeeze = np.squeeze(classes).astype(np.int32)

            keep = py_cpu_nms(boxes_squeeze, scores_squeeze, 0.5)
            boxes_list = np.zeros((100, 4))
            scores_list = np.zeros(100)
            classes_list = np.zeros(100)
            save_nums = len(keep)

            for s in keep:
                # list.index(i)
                boxes_list[keep.index(s), :] = boxes_squeeze[s]
                scores_list[keep.index(s)] = scores_squeeze[s]
                classes_list[keep.index(s)] = classes_squeeze[s]

            classes_list = classes_list.astype(np.int32)
            #因为label编号的问题，所以这里要全部-1
            classes_list_back = classes_list-1

            print(boxes_list)
            print(scores_list)
            print(classes_list)
            #######################################################################
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                boxes_list,
                classes_list_back,
                scores_list,
                category_index,
                use_normalized_coordinates=True,
                min_score_thresh=0.1,
                line_thickness=8)
            return image_np, category_index

def py_cpu_nms(dets, scores, thresh):
    """Pure Python NMS baseline."""
    #x1、y1、x2、y2、以及score赋值
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    #每一个检测框的面积
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # #按照score置信度降序排序
    order = scores.argsort()[::-1]
    keep = []
    # #保留的结果框集合
    while order.size > 0:
        i = order[0]
        keep.append(i)
        #保留该类剩余box中得分最高的一个 #得到相交区域,左上及右下
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        #计算相交的面积,不重叠时面积为0
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        #计算IoU：重叠面积 /（面积1+面积2-重叠面积）
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        #保留IoU小于阈值的box
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
        #因为ovr数组的长度比order数组少一个,所以这里要将所有下标后移一位
    return keep


if __name__ == '__main__':
    start = time.time()
    test_img_path = 'input_dir/00007.jpg'
    np_image, category_index = mainfunction(test_img_path)
    middle = time.time()
    plt.imsave('output_dir/00007.png', np_image)
    end = time.time()
    print(middle - start)
    print(end - middle)