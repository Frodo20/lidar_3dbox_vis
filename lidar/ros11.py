import rospy
import time
import numpy as np
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray

lines = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6],
         [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]


class Detector:
    def __init__(self) -> None:
        rospy.init_node('livox_detector', anonymous=True)

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
        R = self.rotz(heading_angle)
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

    def display(self, boxes):
        self.marker_array.markers.clear()

        for obid in range(len(boxes)):
            ob = boxes[obid]
            tid = 0
            detect_points_set = []
            for i in range(0, 8):
                detect_points_set.append(Point(ob[i], ob[i+8], ob[i+16]))

            marker = Marker()
            marker.header.frame_id = 'livox_frame'
            marker.header.stamp = rospy.Time.now()

            marker.id = obid*2
            marker.action = Marker.ADD
            marker.type = Marker.LINE_LIST

            marker.lifetime = rospy.Duration(0)

            marker.color.r = 1
            marker.color.g = 0
            marker.color.b = 0

            marker.color.a = 1
            marker.scale.x = 0.2
            marker.points = []

            for line in lines:
                marker.points.append(detect_points_set[line[0]])
                marker.points.append(detect_points_set[line[1]])

        self.marker_array.markers.append(marker)

        self.marker_pub.publish(self.marker_array)


if __name__ == '__main__':
    detector = Detector()
    # [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center
    pred_boxes = np.array([[ 1.0289,  0.9812, -0.4731,  0.3622,  0.6111,  1.7270,  1.1762]])
    boxes = []
    for x,y,z,w,l,h,heading in pred_boxes:                                    
        box = detector.get_3d_box((x,y,z),(l,w,h),heading)
        box=box.transpose(1,0).ravel()
        boxes.append(box)
    #print(type(boxes))
    print(boxes)
    for i in range(10):
        print(i)
        detector.display(boxes)

        #time.sleep(1)