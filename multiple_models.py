import numpy as np
import os
import sys
import tensorflow as tf
import cv2
from utils import label_map_util
from utils import visualization_utils as vis_util


cap = cv2.VideoCapture("source_videos/street.mp4")



MODEL_NAME = 'faster_rcnn_inception'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
# PATH_TO_CKPT = "C:/Users/PMac/Anaconda3/Lib/site-packages/tensorflow/models/research/" \
# #                "object_detection/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28/frozen_inference_graph.pb"

PATH_TO_CKPT = "models/faster_rcnn_inception_resnet_v2_atrous_oid_v4_2018_12_12/frozen_inference_graph.pb"
PATH_TO_CKPT_2 = "models/faster_rcnn_nas_coco_2018_01_28/frozen_inference_graph.pb"

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = "labels/oid_v4_label_map.pbtxt"
PATH_TO_LABELS_2 = "labels/mscoco_complete_label_map.pbtxt"

NUM_CLASSES = 601
NUM_CLASSES_2 = 90

img_width = 1280
img_height = 720

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

detection_graph_2 = tf.Graph()
with detection_graph_2.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT_2, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')



label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

label_map_2 = label_map_util.load_labelmap(PATH_TO_LABELS_2)
categories_2 = label_map_util.convert_label_map_to_categories(label_map_2, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index_2 = label_map_util.create_category_index(categories_2)


# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter("resnet_v2_atrous_oid_v4.avi", fourcc, 7.0, (1280,720))

with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
      with detection_graph_2.as_default():
          with tf.Session(graph=detection_graph_2) as sess_2:
            while True:
              ret, image_np = cap.read()

              # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
              image_np_expanded = np.expand_dims(image_np, axis=0)


              image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
              # Each box represents a part of the image where a particular object was detected.
              boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
              # Each score represent how level of confidence for each of the objects.
              # Score is shown on the result image, together with the class label.
              scores = detection_graph.get_tensor_by_name('detection_scores:0')
              classes = detection_graph.get_tensor_by_name('detection_classes:0')
              num_detections = detection_graph.get_tensor_by_name('num_detections:0')
              # Actual detection.
              (boxes, scores, classes, num_detections) = sess.run(
                  [boxes, scores, classes, num_detections],
                  feed_dict={image_tensor: image_np_expanded})

              boxes = np.squeeze(boxes)
              scores = np.squeeze(scores)
              classes = np.squeeze(classes)


              # Visualization of the results of a detection.
              vis_util.visualize_boxes_and_labels_on_image_array(
                  image_np,
                  boxes,
                  classes.astype(np.int32),
                  scores,
                  category_index,
                  use_normalized_coordinates=True,
                  line_thickness=4)

              image_tensor2 = detection_graph_2.get_tensor_by_name('image_tensor:0')
              # Each box represents a part of the image where a particular object was detected.
              boxes2 = detection_graph_2.get_tensor_by_name('detection_boxes:0')
              # Each score represent how level of confidence for each of the objects.
              # Score is shown on the result image, together with the class label.
              scores2 = detection_graph_2.get_tensor_by_name('detection_scores:0')
              classes2 = detection_graph_2.get_tensor_by_name('detection_classes:0')
              num_detections2 = detection_graph_2.get_tensor_by_name('num_detections:0')
              # Actual detection.
              (boxes2, scores2, classes2, num_detections2) = sess_2.run(
                  [boxes2, scores2, classes2, num_detections2],
                  feed_dict={image_tensor2: image_np_expanded})

              boxes2 = np.squeeze(boxes2)
              scores2 = np.squeeze(scores2)
              classes2 = np.squeeze(classes2)

              # Visualization of the results of a detection.
              vis_util.visualize_boxes_and_labels_on_image_array(
                  image_np,
                  boxes2,
                  classes2.astype(np.int32),
                  scores2,
                  category_index_2,
                  use_normalized_coordinates=True,
                  line_thickness=4)


              cv2.imshow('object detection', cv2.resize(image_np, (800,600)))
              #out.write(image_np)


              if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break