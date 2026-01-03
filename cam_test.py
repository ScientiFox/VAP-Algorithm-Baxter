#!/usr/bin/env python

###
#
# Baxter robot materials handling and mixed-autonomy camera test
#  this software tests the eye-in-hand camera on the Baxter robot,
#  and the associated imaging and tracking functionality
#
###

#Standards
import math,time,random

#System-level functions
import sys

#Interface to ROS 
import rospy

#Import the software to talk to Baxter, plus struct, which it uses locally
import baxter_interface
import struct

#message types for ROS
from std_msgs.msg import String
from std_msgs.msg import Header

#Grab camera assets from Baxter package
from baxter_interface.camera import CameraController
from baxter_core_msgs.srv import ListCameras

#Grab a sensor-type ROS message for the images
from sensor_msgs.msg import Image

#Import computer vision assets
import cv,cv2 #Back before cv/cv2 merger
import cv_bridge #package that converts Image ROS type to cv2 Mat type

#Numperical package
import numpy as np

#image processing package from scipy
from scipy import ndimage

class kill_msg:
        #Message to kill the process
	def __init__(self):
		self.kill = False #disable flag

	def kill_cb(self,data):
                #Callback to check termination message
		self.kill = (data.data == 'stop') #Set flag to True on kill message

class cam_callback:
        #Callback to grab images

	def __init__(self):
		#self.cam = _cam
		self.cb_img = -1 #image data member
		self.flag = True #callback activity flag

	def camera_callback(self, data):
		if self.flag: #If the flag is active
			try:
				self.cb_img = data #grab image data if you can
			except:
				self.cb_img = -1 #-1 if it doesn't work
		else: #If not flag set
			pass #donothing


if __name__ == '__main__':

        #Start a ROS node
	rospy.init_node("cam_int_base")

        #Make up the cv type bridge
	bridge = cv_bridge.CvBridge()

        #Try to start the default head camera, then close it- for message loading
	try:
		cam_kill = CameraController("head_camera")
		cam_kill.close()
	except:
		pass

        #Start the eye-in-hand camera
	cam = CameraController("right_hand_camera")
	cam.resolution = (320, 200) #Set the resolution- changes sometimes, need to set here
	cam.open() #start the camera

        #Set the camera callback to the function to the manager object
	cam_callback_var = cam_callback()

        #message handler for stop process message
	kill_var = kill_msg()

        #subscribe the node to the camera feed and the termination feed
	rospy.Subscriber("/cameras/right_hand_camera/image", Image, cam_callback_var.camera_callback)
	rospy.Subscriber("/killcam", String, kill_var.kill_cb)

        #Main loop
	while True:

                #If there's a kill message, end the loop
		if kill_var.kill:
			break

                #If there's an image available
		if cam_callback_var.cb_img != -1:
                        #Make a cv-format image from the Image message
			cv_img = bridge.imgmsg_to_cv2(cam_callback_var.cb_img,"bgr8")

                        #convert to grayscale
			cv_img2 = cv2.cvtColor(cv_img,cv2.COLOR_BGR2GRAY)

                        #noise reduction blur
			cv_img2 = cv2.blur(cv_img2,(8,8))

                        #Threshold the image
			cv_img2 = 255.0*(cv_img2 > 96)

			#Use the blob labeling in ndimage
			labels,n_labels = ndimage.label(cv_img2)

                        #Plot the labeled image by color
			cv_img[:,:,0] = labels*(255.0/n_labels)
			cv_img[:,:,1] = 0
			cv_img[:,:,2] = 255.0 - labels*(255.0/n_labels)

                        #Optional save for each frame
			#cv2.imwrite("[put a location]/data.jpg",cv_img)

                        #Show the output
			cv2.imshow("img_fetch",cv_img)
			cv2.waitKey(1)

                #Smoothness delay
		time.sleep(0.3)

	

