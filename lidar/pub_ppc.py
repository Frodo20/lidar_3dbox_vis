#from skimage.transform import resize as imresize
#from imageio import imread
import os
import numpy as np
#from pathlib import Path
import PIL.Image as pil
import scipy.misc
from PIL import Image as imge
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs import point_cloud2
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
bridge = CvBridge()
i = 0

from visualization_msgs.msg import Marker, MarkerArray
import os

lines = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6],
         [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]


class Detector:
    def __init__(self):
        # rospy.init_node('livox_detector', anonymous=True)

        self.marker_pub = rospy.Publisher(
            '/detect_box3d', MarkerArray, queue_size=2)

        self.marker_array = MarkerArray()

    def rotx(self, t):
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[1,  0,  0],
                        [0,  c,  -s],
                        [0, s,  c]])
    def roty(self, t):
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c,  0,  s],
                        [0,  1,  0],
                        [-s, 0,  c]])

    def rotz(self,t):
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c,  -s,  0],
                        [s,  c,  0],
                        [0, 0,  1]])

    def get_3d_box(self, center, box_size, heading_angle):
        ''' Calculate 3D bounding box corners from its parameterization.

        Input:heading_angle
            box_size: tuple of (l,w,h)
            : rad scalar, clockwise from pos z axis
            center: tuple of (x,y,z)
        Output:
            corners_3d: numpy array of shape (8,3) for 3D box cornders
        '''
        R = self.roty(heading_angle)
        # R = np.eye(3)
        l, w, h = box_size
        x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
        y_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
        z_corners = [h/2, h/2, h/2, h/2, -h/2, -h/2, -h/2, -h/2]
        corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
        corners_3d[0, :] = corners_3d[0, :] + center[0]
        corners_3d[1, :] = corners_3d[1, :] + center[1]
        corners_3d[2, :] = corners_3d[2, :] + center[2]
        corners_3d = np.transpose(corners_3d)
        return corners_3d

    def display(self, boxes, ii):
        
        # print('--------', boxes)
        for obid in range(len(boxes)):
            ob = boxes[obid]
            print('******************:',ob)
            tid = 0
            detect_points_set = []
            for i in range(0, 8):
                detect_points_set.append(Point(ob[i], ob[i+8], ob[i+16]))

            marker = Marker()
            marker.header.frame_id = 'velo_link'
            marker.header.stamp = rospy.Time.now()

            marker.id = ii
            marker.action = Marker.ADD
            marker.type = Marker.LINE_LIST

            marker.lifetime = rospy.Duration(0)

            marker.color.r = 0.98
            marker.color.g = 0.77
            marker.color.b = 0.79
            marker.color.a = 0.75 #alpha
            
            marker.scale.x = 0.15
            marker.scale.y = 0.1
            marker.scale.z = 0.1

            marker.points = []

            for line in lines:
                marker.points.append(detect_points_set[line[0]])
                marker.points.append(detect_points_set[line[1]])
       
        self.marker_array.markers.append(marker)

        self.marker_pub.publish(self.marker_array)
        self.marker_array.markers *= 0

cam_pub = rospy.Publisher('kitti_cam', Image,queue_size=2)
cam_pubXYZ = rospy.Publisher('XYZMap', Image,queue_size=2)
cam_pubuv = rospy.Publisher('uvMap', Image,queue_size=2)
cam_pubproj = rospy.Publisher('proj_velo2cam', Image,queue_size=2)
cam_pubimage3d = rospy.Publisher('image_box3d', Image,queue_size=2)
lidarpc_pub = rospy.Publisher("LiDARPC", PointCloud2, queue_size=2)

def generate_pointcloud1(ii):
    lidar_path = 'raw_lidar/%06d.npy'%ii
    # lidar_path = os.path.join('/dataset/data_odometry_velodyne/00/', "velodyne/%06d.bin"%ii)
    with open(lidar_path, 'rb') as fp1:
        # scan = np.fromfile(fp1, dtype=np.float32).reshape((-1,4))
        # pc1 = scan[:, 0:3] # lidar xyz (front, left, up)
        pc1 = np.load(fp1)
        z = np.zeros(pc1.shape[0]).reshape(pc1.shape[0],1)
        rgb = np.ones(pc1.shape[0]).reshape(pc1.shape[0],1)
        points1 = np.concatenate((pc1,rgb,z,z),axis=1)

    header = Header()
    #header.frame_id = "try_pointcloud"
    header.frame_id = "velo_link"
    fields = [
                PointField('x', 0, PointField.FLOAT32, 1),
                PointField('y', 4, PointField.FLOAT32, 1),
                PointField('z', 8, PointField.FLOAT32, 1),
                PointField('r', 12, PointField.FLOAT32, 1),
                PointField('g', 16, PointField.FLOAT32, 1),
                PointField('b', 20, PointField.FLOAT32, 1)]
    pointcloud1 = point_cloud2.create_cloud(header=header,fields=fields, points=points1)
    return pointcloud1


def image_callback(msg):
    detector = Detector()
    global i
    global cam_pub, cam_pubXYZ, cam_pubproj, cam_pubimage3d, lidarpc_pub
    i = i + 1
    print("Received an image!/%06d"%i)
    
    DATA_PATH = './'
    img = cv2.imread(os.path.join('/dataset/data_odometry_color/00/', 'image_2/%06d.png'%i))
    cam_pub.publish(bridge.cv2_to_imgmsg(img,"bgr8"))

    XYZMap = cv2.imread(os.path.join(DATA_PATH, 'XYZ_Map/%06d.png'%i))
    cam_pubXYZ.publish(bridge.cv2_to_imgmsg(XYZMap,"bgr8"))

    uvMap = cv2.imread(os.path.join(DATA_PATH, 'uv_Map/%06d.png'%i))
    cam_pubuv.publish(bridge.cv2_to_imgmsg(uvMap,"bgr8"))

    proj_image = cv2.imread(os.path.join(DATA_PATH, 'velo2cam/%06d.png'%i))
    cam_pubproj.publish(bridge.cv2_to_imgmsg(proj_image,"bgr8"))

    image_box = cv2.imread(os.path.join(DATA_PATH, 'image_box/%06d.png'%i))
    cam_pubimage3d.publish(bridge.cv2_to_imgmsg(image_box,"bgr8"))

    lidarpc= generate_pointcloud1(i)
    
    lidarpc_pub.publish(lidarpc)

    pred_file = os.path.join('./', "pred_box/%06d.txt"%i)
    with open(pred_file,'r') as fp:
        pred_lines = fp.readlines()
    boxes3d_corner = []
    num_obj = len(pred_lines)
    boxes = []
    for j in range(num_obj):
        obj = pred_lines[j].strip().split(' ')
        obj_class = obj[0].strip()
        if obj_class not in ['Car']:
            continue
    
        pred_boxes = np.array([[ float(obj[11]), float(obj[12]), float(obj[13]), float(obj[8]), float(obj[9]), float(obj[10]), float(obj[14])]])
        
        for x,y,z,w,l,h,heading in pred_boxes:
            print('x,y,z,w,l,h,heading:',x,y,z,w,l,h,heading)
            box = detector.get_3d_box((x,y,z),(l,w,h),heading)
            box=box.transpose(1,0).ravel()
            print('box:',box,box.shape)
            boxes.append(box)
    for j in range(num_obj):
        boxes1 = [np.array(boxes[j])]
        detector.display(boxes1, j)


def main():
    
    rospy.init_node('image_listener')
    image_topic = "/kitti/camera_color_left/image_raw"
    rospy.Subscriber(image_topic, Image, image_callback)
    rospy.spin()

if __name__ == '__main__':
    main()
