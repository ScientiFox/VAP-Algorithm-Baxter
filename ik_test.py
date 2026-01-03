#!/usr/bin/env python

###
#
# Baxter robot materials handling and mixed-autonomy inverse
#  kinematics test. This software tests the Baxter IK package
#  and communication and calculation to the robot
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

#Messages for geometric values from robot
from geometry_msgs.msg import (
    PoseStamped,
    Pose,
    Point,
    Quaternion,
)

#message types for ROS
from std_msgs.msg import Header

#baxter IK messages
from baxter_core_msgs.srv import (
    SolvePositionIK,
    SolvePositionIKRequest,
)

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

#Function to calculate the inverse kinematics from positions and angles
def get_IK(limb,px,py,pz,ox,oy,oz,ow):

        #construct the base message
	ns = "ExternalTools/" + limb + "/PositionKinematicsNode/IKService"

        #Fire off the IK solver message to the solver service
	iksvc = rospy.ServiceProxy(ns, SolvePositionIK)
	ikreq = SolvePositionIKRequest()

        #build a ROSpy header- you just have to have it for the message
	hdr = Header(stamp=rospy.Time.now(), frame_id='base')

	# Make a pose msg:
	pos = Point(x=px,y=py,z=pz) #Make a point from coords
	orient = Quaternion(x=ox,y=oy,z=oz,w=ow) #Make quaternion orientation
	_pose = Pose(pos,orient) #Pose is both of those together
	pstamp = PoseStamped(header=hdr,pose=_pose) #Pair the header and pose together

        #Add the stamped pose to the request
	ikreq.pose_stamp.append(pstamp)

        #Try and get the response
	try:
		rospy.wait_for_service(ns, 5.0) #Give the request up to 5s (average is ~2.5s)
		resp = iksvc(ikreq) #grab the response
	except (rospy.ServiceException, rospy.ROSException), e: #timeout failure
		rospy.logerr("Service call failed: %s" % (e,)) #Log the error and end the function
		return True

        #Convert the IK message
	resp_seeds = struct.unpack('<%dB' % len(resp.result_type),resp.result_type)

        #Check if the result is a valid IK solution
	if (resp_seeds[0] != resp.RESULT_INVALID):
		# Format solution into Limb API-compatible dictionary
		limb_joints = dict(zip(resp.joints[0].name, resp.joints[0].position))
		return limb_joints #return the answer
	else: #If not valid, report and return False
		print("No Valid Joint Solution Found.")
		return False

class callback:
        # callback for coordinate messages

	def __init__(self):
                #Init with all -1s on startup
		self.cb_list = [-1,-1,-1,-1,-1,-1]

	def cb(self,data):
                #When message is called

		if data.data != 'stop': #if not a stop message
			coords = data.data #grab the message data
			coords = coords.split(",") #break up text by , delimiter
			coords = [float(a) for a in coords] #Make the listed coord strings to floats
		else: #otherwise, annotate stop condition
			coords = ['stop']

                #Update the coordinate variable
		self.cb_list = coords

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

def rpy_to_wxyz(r,p,y):
        #Helper function to convert RPY angles to quaternions

        #Construct W from angles
	owr = math.cos(r/2.0)*math.cos(p/2.0)*math.cos(y/2.0)
	owr = owr + math.sin(r/2.0)*math.sin(p/2.0)*math.sin(y/2.0)

        #Construct X from angles
	oxr = math.sin(r/2.0)*math.cos(p/2.0)*math.cos(y/2.0)
	oxr = oxr - math.cos(r/2.0)*math.sin(p/2.0)*math.sin(y/2.0)

        #Construct Y from angles
	oyr = math.cos(r/2.0)*math.sin(p/2.0)*math.cos(y/2.0)
	oyr = oyr + math.sin(r/2.0)*math.cos(p/2.0)*math.sin(y/2.0)

        #Construct Z from angles
	ozr = math.cos(r/2.0)*math.cos(p/2.0)*math.sin(y/2.0)
	ozr = ozr - math.sin(r/2.0)*math.sin(p/2.0)*math.cos(y/2.0)

        #return the quaternions
	return owr,oxr,oyr,ozr

if __name__ == '__main__':
        #Main function

        #Make up the cv type bridge
	bridge = cv_bridge.CvBridge()

        #Start IK node
	print "starting node"
	rospy.init_node("rsdk_ik_to_limbs")

        #Make the callback handler
	callback_var = callback()

        #Swap the cameras
	print "changing cameras"
	try: #When the head camera is active to be disabled-
		cam_kill = CameraController("head_camera") #Start and kill the head camera
		cam_kill.close()
	except: #Otherwise, do nothing
		pass

        #Turn on the right hand camera
	print "activating right camera"
	cam = CameraController("right_hand_camera")
	cam.resolution = (320, 200) #Set resolution
	cam.fps = 3 #set FPS
	cam.gain = -1 #set gain
	cam.window = (530,350) #set display window
	cam.open() #start camera

        #Start the camera handler object
	cam_callback_var = cam_callback()

        #Subscribe to hand image message and coordinate messages
	print "Setting subscriptions"
	cam_sub = rospy.Subscriber("/cameras/right_hand_camera/image", Image, cam_callback_var.camera_callback)
	rospy.Subscriber("/coords", String, callback_var.cb)

        #Activate the arm controller
	print "initializing arm"
	right = baxter_interface.Limb('right')

        #Activate the gripper controller
	grip_right = baxter_interface.Gripper('right', baxter_interface.CHECK_VERSION)

        #On a gripper error, reset it
	if grip_right.error():
		grip_right.reset()

        #if the gripper isn't calibrated, send calibration command
	if (not grip_right.calibrated() and 
		grip_right.type() != 'custom'):
		grip_right.calibrate()

	#A sequences of poses to aproach and retreat from the workspace
	print "Workspace clearance movements"
	aS_0 = [0.1,-0.7,-0.4, 0.0,-3.142,0.0]
	aS_1 = [0.1,-0.7,0.2, 0.0,-3.142,0.0]
	aS_2 = [0.7,-0.4,0.4, 0.0,-3.142,0.0]
	closeout_poses = [aS_0,aS_1,aS_2]

        #Execute each of the three approach maneuvers in order
	for i in [0,1,2]:
                #Announce position index
		print "move to position: "+ str(i)
		pos = closeout_poses[i] #Grab position target
		pxr = pos[0] #grab physical pose position
		pyr = pos[1]
		pzr = pos[2]
		owr,oxr,oyr,ozr = rpy_to_wxyz(pos[3],pos[4],pos[5]) #Make quaternions from RPY
		r_joints = get_IK('right',pxr,pyr,pzr,oxr,oyr,ozr,owr) #Get the IK for this position
		cam_sub.unregister() #Disconnect camera- have to for moves, not sure why.
		right.move_to_joint_positions(r_joints) #Send move command

                #Reconnect to camera
		cam_sub = rospy.Subscriber("/cameras/right_hand_camera/image", Image, cam_callback_var.camera_callback)

	# Initial workspace pose
	pxr_o = 0.7
	pyr_o = -0.4
	pzr_o = 0.4

        #Put init pose into variables
	pxr = pxr_o
	pyr = pyr_o
	pzr = pzr_o


	#Vertical down: yaw controls rotation of gripper
	#Left-facing
	RPY_canon = [[0.0,-1.0*math.pi,0.0],[0.5*math.pi,0.0,0.0]]

        #Set a standard angle
	r = -0.0*math.pi
	p = -1.0*math.pi
	y = 0.0*math.pi

        #Make up quaternions
	owr,oxr,oyr,ozr = rpy_to_wxyz(r,p,y)
	pos_p = [-1,-1,-1,-1,-1,-1] #position command holder

        #set gripper state flag
	r_grip_state = 0

        #Test indices
	i = 2
	j = 2
	n = 0

        #Start the main test loop
	print "starting loop"
	while True:

                #After initial counter
		if n != 0:
			pos = callback_var.cb_list #grab the coordinate list from the message callback
		else: #On the initial counter
			pos = [pxr,pyr,pzr,r,p,y] #set to defailt start pose

                #If a stop command issued and not on the first step
		if pos[0] == 'stop' and n != 0:
			for i in [2,1,0]: #Run through the closeout poses in reverse order
				pos = closeout_poses[i]
				pxr = pos[0]
				pyr = pos[1]
				pzr = pos[2]
				owr,oxr,oyr,ozr = rpy_to_wxyz(pos[3],pos[4],pos[5])
				r_joints = get_IK('right',pxr,pyr,pzr,oxr,oyr,ozr,owr)
				cam_sub.unregister()
				right.move_to_joint_positions(r_joints)
				cam_sub = rospy.Subscriber("/cameras/right_hand_camera/image", Image, cam_callback_var.camera_callback)
				time.sleep(0.3)
			break #End the loop

                #if an open/close message, change gripper state
		elif pos[0] == "G_open":
			r_grip_state = 0
		elif pos[0] == "G_close":
			r_grip_state = 1

                #if a 6-long pose coordinate list
		elif len(pos) == 6:
                        #Check if the pose is at least 1cm malahanois away from most recent pose point
			if sum([abs(pos[i]-pos_p[i]) for i in range(6)])>0.01:
				pxr = pos[0] #Grab coords
				pyr = pos[1]
				pzr = pos[2]
				owr,oxr,oyr,ozr = rpy_to_wxyz(pos[3],pos[4],pos[5]) #transform RPY to quaternion
				pos_p = pos+[] #update prior pose to new pose
			elif n != 0:
				pos = pos + [-1] #annotate if not enough movement

                #If a real and valid movement
		if not(-1 in pos):
                        #Construct the IK for the new pose
			r_joints = get_IK('right',pxr,pyr,pzr,oxr,oyr,ozr,owr)
		else: #Otherwise, no joint angles
			r_joints = False

                #if new joint positions available
		if r_joints != False:
			try: #Try a movement
				cam_sub.unregister() #Disconnect camera- not sure why but it's necessary
				right.move_to_joint_positions(r_joints) #Send joint positions

				#Resubscribe to camera
				cam_sub = rospy.Subscriber("/cameras/right_hand_camera/image", Image, cam_callback_var.camera_callback)
			except: #Otherwise, donothing
				pass

                        #Send gripper command for current state
			if r_grip_state == 0:
				grip_right.open()
			elif r_grip_state == 1:
				grip_right.close()

                #Optional diagnostic output of coords
		#print n,pxr,pyr,pzr,oxr,oyr,ozr,owr

                #If camera data is available
		if cam_callback_var.cb_img != -1:
                        #convert to cv type- an original and one for analysis
			img_o = bridge.imgmsg_to_cv2(cam_callback_var.cb_img,"bgr8")
			cv_img = bridge.imgmsg_to_cv2(cam_callback_var.cb_img,"bgr8")

                        #SMoothing blur, then convert to grayscale
			cv_img2 = cv2.blur(cv_img,(4,4))
			cv_img2 = cv2.cvtColor(cv_img2,cv2.COLOR_BGR2GRAY)

                        # Optional histogram analysis clasifier- superceded
			#hist,bins = np.histogram(cv_img2.flatten(),256,[0,256])
			#cdf = hist.cumsum()
			#cdf_m = np.ma.masked_equal(cdf,0)
			#cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
			#cdf = np.ma.filled(cdf_m,0).astype('uint8')
			#cv_img2 = cdf[cv_img2]

                        #level and threshold image, then blob label the resulting regions
			cv_img2 = 16.0*(cv_img2 < 3.0*np.average(cv_img2)/4.0)
			labels,n_labels = ndimage.label(cv_img2)

                        #coordinate meshgrid for the image
			x = np.linspace(0, 319, 320)
			y = np.linspace(0, 199, 200)
			xv, yv = np.meshgrid(x, y)

			thetas = [] #holder for orientaiton angles

                        #for each label in the blob set
			for i in range(n_labels+1):
				filt = 1.0*(labels==i) #filter that label's blob
				area = np.sum(filt) #get the number of pixels the blob fills

                                #If more than 100px area, and less than 20% of full field
				if area > 100 and area < 0.2*(200*320):
					x_set = filt*xv #Get x and y coords from meshgrid and filter
					y_set = filt*yv

                                        #grab bounding box as non-zero min and max in coords
					bb_xmin = int(np.min(np.min(x_set+(1000*(x_set==0)))))
					bb_ymin = int(np.min(np.min(y_set+(1000*(x_set==0)))))
					bb_xmax = int(np.max(np.max(x_set)))
					bb_ymax = int(np.max(np.max(y_set)))

                                        #grab image area within the bounding box
					img_o[bb_ymax-1:bb_ymax+1,bb_xmin-1:bb_xmin+1,1] = 255.0
					img_o[bb_ymax-1:bb_ymax+1,bb_xmax-1:bb_xmax+1,1] = 255.0
					img_o[bb_ymin-1:bb_ymin+1,bb_xmin-1:bb_xmin+1,1] = 255.0
					img_o[bb_ymin-1:bb_ymin+1,bb_xmax-1:bb_xmax+1,1] = 255.0

                                        #calculate center of mass
					com_x = int(np.sum(x_set)/area)
					com_y = int(np.sum(y_set)/area)

                                        #get bounding box area
					a_bb = 1.0*(bb_xmax-bb_xmin)*(bb_ymax-bb_ymin)

					#Calculate orientation angle approximation
					theta = 2.0*math.asin((area/a_bb - 1.0))+0.262

                                        #Add to angle list
					thetas = thetas + [round(theta*180.0/(math.pi))]

                                        #mark COM with a red dot
					img_o[com_y-5:com_y+5,com_x-5:com_x+5,2] = 255.0
					img_o[com_y-5:com_y+5,com_x-5:com_x+5,1] = 0.0
					img_o[com_y-5:com_y+5,com_x-5:com_x+5,0] = 0.0

                                        #orientation vector corners
					v1 = [(bb_xmax+bb_xmin)/2-com_x,bb_ymin-com_y]
					v2 = [(bb_xmax+bb_xmin)/2-com_x,bb_ymax-com_y]

                                        #Calculate rotated orientation vectors
					v1r = [int(math.cos(theta)*v1[0]-math.sin(theta)*v1[1]),
								int(math.sin(theta)*v1[0]+math.cos(theta)*v1[1])]
					v2r = [int(math.cos(theta)*v2[0]-math.sin(theta)*v2[1]),
								int(math.sin(theta)*v2[0]+math.cos(theta)*v2[1])]

                                        #next rotation angle and vectors
					theta2 = theta + math.pi/2.0
					v3r = [int(math.cos(theta2)*v1[0]-math.sin(theta2)*v1[1]),
								int(math.sin(theta2)*v1[0]+math.cos(theta2)*v1[1])]
					v4r = [int(math.cos(theta2)*v2[0]-math.sin(theta2)*v2[1]),
								int(math.sin(theta2)*v2[0]+math.cos(theta2)*v2[1])]

                                        #Plot orientation vector box
					cv2.line(img_o,(com_x+v1r[0],com_y+v1r[1]),(com_x,com_y),(0,255,0),1)
					cv2.line(img_o,(com_x+v2r[0],com_y+v2r[1]),(com_x,com_y),(0,255,0),1)
					cv2.line(img_o,(com_x+v3r[0],com_y+v3r[1]),(com_x,com_y),(0,255,0),1)
					cv2.line(img_o,(com_x+v4r[0],com_y+v4r[1]),(com_x,com_y),(0,255,0),1)

                        #Optional blob index display, color-coded blobs
			#cv_img[:,:,0] = labels*(255.0/(n_labels+1))
			#cv_img[:,:,1] = 0
			#cv_img[:,:,2] = 255.0 - labels*(255.0/(n_labels+1))
			#cv2.imshow("img_fetch",filt*xv)

                        #List of angles found
			print(thetas)

                        #Show diagnostic images
			cv2.imshow("img_fetch",img_o*1.0)
			cv2.waitKey(1)

                #While waiting on image data
		else:
			print "almost..."

                #Increment imaging counter
		n+=1

                #Optional smoothness delay
		#time.sleep(0.1)



