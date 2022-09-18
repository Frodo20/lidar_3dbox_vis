#!/usr/bin/env python


import os
from data_utils import *
from publish_utils import *
from kitti_utils import *

DAtA_PATH = '/home/frodo/Downloads/lidar'

def compute_3d_box_cam2(h,w,l,x,y,z,yaw):
    R = np.array([[np.cos(yaw),0,np.sin(yaw)],[0,1,0],[-np.sin(yaw),0,np.cos(yaw)]])
    x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2]
    y_corners = [0,0,0,0,-h,-h,-h,-h]
    z_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2]
    corners_3d_cam2 = np.dot(R, np.vstack([x_corners,y_corners,z_corners]))
    corners_3d_cam2 += np.vstack([x,y,z])
    return corners_3d_cam2


if __name__ == '__main__':
    frame = 0
    rospy.init_node('kitty_node',anonymous=True)
    #cam_pub = rospy.Publisher('kitti_cam',Image,queue_size=10)
    pcl_pub = rospy.Publisher('kitti_point_cloud',PointCloud2,queue_size=10)
    #ego_pub = rospy.Publisher('kitti_ego_car',MarkerArray,queue_size=10)
    #imu_pub = rospy.Publisher('kitti_imu',Imu,queue_size=10)
    #gps_pub = rospy.Publisher('kitti_gps',NavSatFix,queue_size=10)
    box3d_pub = rospy.Publisher('kitti_3d',MarkerArray,queue_size=10)
    #model_pub = rospy.Publisher('kitti_car_model',Marker,queue_size=10)
    bridge = CvBridge()
	
	
    rate = rospy.Rate(10)
    
    #df_tracking = read_tracking('/home/wsj/data/kitty/training/label_02/0000.txt')
    #df_tracking = read_tracking('/home/frodo/Downloads/lidar/label_02/pred_box/%06d.txt'%frame)
    
    calib = Calibration('/home/frodo/Downloads/lidar/calib/',from_video=True)
    
    while not rospy.is_shutdown():
        
        df_tracking_frame = read_tracking('/home/frodo/Downloads/lidar/pred_box/%06d.txt'%frame)
        types = np.array(df_tracking_frame['type'])
        #df_tracking_frame = df_tracking[df_tracking.frame==frame]
    	#读取2D检测框数据
        #boxes_2d = np.array(df_tracking_frame[['bbox_left', 'bbox_top', 'bbox_right', 'bbox_bottom']])
        #types = np.array(df_tracking_frame['type'])
        #读取摄像头数据
        #image = read_camera(os.path.join(DAtA_PATH, 'image_02/data/%010d.png'%frame))
        #使用np读取点云 
        point_cloud = read_point_cloud(os.path.join(DAtA_PATH, 'velo/%06d.bin'%frame))
        
        
        #读取3D检测框数据
        boxes_3d = np.array(df_tracking_frame[['height', 'width', 'length', 'pos_x', 'pos_y', 'pos_z', 'rot_y']])
        
        corners_3d_velos = []
        
        for box_3d in boxes_3d:
            corners_3d_cam2 = compute_3d_box_cam2(*box_3d)
            #corners_3d_velo = calib.project_rect_to_velo(corners_3d_cam2.T)
            #corners_3d_velos += [corners_3d_velo]
            corners_3d_velos += [corners_3d_cam2]
        """
        for box_3d in boxes_3d:
            corner_3d_cam2 = compute_3d_box_cam2(*box_3d)
            corner_3d_velo = calib.project_rect_to_velo(np.array(corner_3d_cam2).T)
            corner_3d_velos += [corner_3d_velo]
        """
        #发布摄像头和点云数据
        #publish_camera(cam_pub,bridge,image,boxes_2d,types)
        publish_point_cloud(pcl_pub,point_cloud)
        
        #发布ego_car
        #publish_ego_car(ego_pub)
        
        
        #发布汽车模型
        #publish_car_model(model_pub)
        
        #读取imu和gps数据
        #imu_data = read_imu(os.path.join(DAtA_PATH, 'oxts/data/%010d.txt'%frame))
        
        
        #发布imu数据
        #publish_imu(imu_pub,imu_data)
        
        
        #发布gps数据
        #publish_gps(gps_pub,imu_data)
        
         #发布3D框数据
        publish_3dbox(box3d_pub,corners_3d_velos,types)
        
        rospy.loginfo("published...")
        rate.sleep()
        frame += 1
        frame %= 11
        
        
       






