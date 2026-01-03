#!/usr/bin/env python

###
#
# Baxter robot materials handling and mixed-autonomy
#  This software is the main operational loop for running
#  standard experiments and demonstrations
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

#Import joystick interface library
import joystick

#message types for ROS
from std_msgs.msg import Header
from std_msgs.msg import String
from sensor_msgs.msg import Range

from geometry_msgs.msg import (
    PoseStamped,
    Pose,
    Point,
    Quaternion,
)

#baxter IK messages
from baxter_core_msgs.srv import (
    SolvePositionIK,
    SolvePositionIKRequest,
)
from baxter_interface.camera import CameraController
from baxter_core_msgs.srv import ListCameras
from sensor_msgs.msg import Image

#Import computer vision assets
import cv,cv2 #Back before cv/cv2 merger
import cv_bridge #package that converts Image ROS type to cv2 Mat type

#Numperical package
import numpy as np

#image processing package from scipy
from scipy import ndimage

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

class ir_callback:
        #Callback for IR in-hand distance sensor messages

	def __init__(self):
                #Start with no range data
		self.range = -1

	def ir_cb(self,data):
                #If data available,
		if data.range < 60: #Save actual ranging data up to 60cm range
			self.range = data.range
		else: #flagged if over detection range
			self.range = 9999

class cam_callback:
        #Callback to grab images

	def __init__(self):
		self.cb_img = -1 #image data member
		self.flag = True #callback activity flag
		self.ct = 0 #sequence counter

	def camera_callback(self, data):
		if self.flag: #If the flag is active
			try:
				self.cb_img = data #grab image data if you can
				self.ct = self.ct + 1 #Increase counter if you do
			except:
				self.cb_img = -1
		else: #If not flag set
			pass #donothing

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

def img_segment(bkg_img, img_o,hist_labeled):
        #Image segmentation routine

        #Difference image filter
	img_d = np.abs(bkg_img - img_o)

	img_d[0:40,0:100,:] = 0.0 #Extract out-of-field corners
	img_d[0:40,-100:,:] = 0.0

        #Blur and threshold difference image, and make thresholded intensity sum
	img_d = cv2.blur(img_d,(5,5))
	img_d_0 = (img_d[:,:,0]>70.0)*1.0
	img_d_1 = (img_d[:,:,1]>70.0)*1.0
	img_d_2 = (img_d[:,:,2]>70.0)*1.0
	img_q = 1.0*((img_d_0+img_d_1+img_d_2)>0.5)

        #Label prior binary image
	labels,n_labels = ndimage.label(img_q)

        #Make an empty copy of the original
	img_cpy = img_o + 0
	img_o[:,:,0] = 0.0
	img_o[:,:,1] = 0.0
	img_o[:,:,2] = 0.0

        #Coodrinate grid array
	x = np.linspace(0, 319, 320)
	y = np.linspace(0, 199, 200)
	xv, yv = np.meshgrid(x, y)

        #Maximum field proportions
	Y_MAX = 0.91
	Y_MIN = 0.49
	X_MAX = 0.73
	X_MIN = 0.08

        #Object pose properties
	thetas = []
	obj_ids = []
	locs = []

        #Checking each labeled category in image
	for i in range(n_labels+1):

                #Select this label region
		filt = 1.0*(labels==i)

                #Get this blob's area
		area = np.sum(filt)

                #Filter for areas between 225 minimum pixels, and less than 20% of total area
		if area > 225 and area < 0.2*(200*320):
			x_set = filt*xv #Get x and y coords in the blob
			y_set = filt*yv

                        #Calculate object bounding box
			bb_xmin = int(np.min(np.min(x_set+(1000*(x_set==0)))))
			bb_ymin = int(np.min(np.min(y_set+(1000*(x_set==0)))))
			bb_xmax = int(np.max(np.max(x_set)))
			bb_ymax = int(np.max(np.max(y_set)))

                        #get upper left and right region coord sums in bounding box area
			tlt_ul = np.sum(filt[bb_ymin:(bb_ymin+bb_ymax)/2,
									bb_xmin:(bb_xmin+bb_xmax)/2])
			tlt_ur = np.sum(filt[(bb_ymin+bb_ymax)/2:bb_ymax,
									bb_xmin:(bb_xmin+bb_xmax)/2])

                        #set region highlighting on output image
			img_o[bb_ymin:(bb_ymin+bb_ymax)/2,bb_xmin:(bb_xmin+bb_xmax)/2,1] = 64.0
			img_o[(bb_ymin+bb_ymax)/2:bb_ymax,bb_xmin:(bb_xmin+bb_xmax)/2,1] = 128.0
			img_o[bb_ymax-1:bb_ymax+1,bb_xmin-1:bb_xmin+1,1] = 255.0
			img_o[bb_ymax-1:bb_ymax+1,bb_xmax-1:bb_xmax+1,1] = 255.0
			img_o[bb_ymin-1:bb_ymin+1,bb_xmin-1:bb_xmin+1,1] = 255.0
			img_o[bb_ymin-1:bb_ymin+1,bb_xmax-1:bb_xmax+1,1] = 255.0

                        #Calculate zone COM coords
			com_x = int(np.sum(x_set)/area)
			com_y = int(np.sum(y_set)/area)

                        #bounding box area
			a_bb = (1.0*bb_xmax-bb_xmin)*(bb_ymax-bb_ymin)

                        #calculate tilt orientation and aspect ratio
			tlt = 1.0*(tlt_ul > tlt_ur)
			asp_r = (1.0*bb_xmax-bb_xmin)/(bb_ymax-bb_ymin)

                        #cases for generally extreme vases of very small or large aspect ratio or region to box area
			if (area/a_bb > 0.75) or (0.5 > asp_r) or (asp_r > 2.0):
				if (bb_xmax-bb_xmin) < (bb_ymax-bb_ymin):
					theta = 0.0 #If horizontal- 0 degrees
				else:
					theta = 1.57 #if vertical-  90 degrees
			else: #Otherwise:
				if tlt: #if tilted up, angle by inverse tangent of area vs positive x
					theta = math.atan((1.0*bb_ymax-bb_ymin)/(bb_xmax-bb_xmin))
				else: #if tilted down, angle by inverse tangent of area vs negative x
					theta = math.atan((1.0*bb_ymax-bb_ymin)/(bb_xmin-bb_xmax))

                        #orientation vector and orthogonal vector
			v2 = [(bb_xmax+bb_xmin)/2-com_x,bb_ymax-com_y]
			v2r = [int(math.cos(theta)*v2[0]-math.sin(theta)*v2[1]),
						int(math.sin(theta)*v2[0]+math.cos(theta)*v2[1])]

                        #add angles to list
			thetas = thetas + [(round(area/a_bb,2),round(theta,3))]

                        #Mark output image with blue for upwards tilt and violet for down
			if tlt:
				img_o[:,:,0] = img_o[:,:,0] + 255.0*filt
			else:
				img_o[:,:,0] = img_o[:,:,0] + 128.0*filt
				img_o[:,:,2] = img_o[:,:,2] + 128.0*filt

                        #Mark COM of region on output image
			img_o[com_y-5:com_y+5,com_x-5:com_x+5,2] = 255.0
			img_o[com_y-5:com_y+5,com_x-5:com_x+5,1] = 0.0
			img_o[com_y-5:com_y+5,com_x-5:com_x+5,0] = 0.0

			#Copy original image and select out filtered color levels in region
			img_cpy2 = img_cpy*0
			img_cpy2[:,:,0] = img_cpy[:,:,0]*filt
			img_cpy2[:,:,1] = img_cpy[:,:,1]*filt
			img_cpy2[:,:,2] = img_cpy[:,:,2]*filt

                        #Calculate histograms in each color channel
			histb = cv2.calcHist([img_cpy2],[0],None,[32],[0,256])
			histg = cv2.calcHist([img_cpy2],[1],None,[32],[0,256])
			histr = cv2.calcHist([img_cpy2],[2],None,[32],[0,256])

                        #Color histogram comparison
			d_min = -1 #Minimum histogram vector distance
			obj_cands = [] #candidate object list

			#Looping over labeled histogram vectors for known objects
			for a in hist_labeled:
                                #Calculate the magnitude vector distance components
				d_hist = np.abs(histb - a[1][0]) + np.abs(histg - a[1][1]) + np.abs(histr - a[1][2])
				d_hist = np.sum(d_hist) #sum up all differences
				obj_cands = obj_cands + [(a[0],d_hist)] #add label and distance for each item type
			obj_cands.sort(key=lambda x: x[1]) #Sort objects by correlation vector magnitude
			obj_ids = obj_ids + [obj_cands[0]] #Grab the lowest distance match for the selected type

                        #Draw a line marking the COM of the current item, and label it at that line
			cv2.line(img_o,(com_x+v2r[0],com_y+v2r[1]),(com_x,com_y),(0,255,0),1)
			font = cv2.FONT_HERSHEY_SIMPLEX
			cv2.putText(img_o,obj_cands[0][0],(com_x-5,com_y-10), font, 
									0.5,(0,255,255),1,cv.CV_AA)

                        #Calibration offsets (from single-point manual positioning tests)
			offs_x = 0.4-0.003 #0.72->0.7 ; 0.585->0.6 ; 0.403 -> 0.4
			offs_y = 0.7-0.009 #0.21->0.22 ; 0.629->0.6 ; 0.707 -> 0.7
			c_offy = -0.07
			c_offx = -0.05
			alpha = 0.83

                        #Calculated item locations scaled to imaging plane frame and offset calibrations
			loc = (alpha*(1.0-(com_y)/100.0)*((Y_MAX-Y_MIN)/2.0)+offs_y+c_offy,
						alpha*(1.0-(com_x)/160.0)*((X_MAX-X_MIN)/2.0)+offs_x+c_offx)
			locs = locs + [(round(loc[0],3),round(loc[1],3))] #Add calibrated pose to list

        #Return marked image, object angles, ids, and locations
	return img_o,thetas,labels,obj_ids,locs

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

# Primary execution Loop
if __name__ == '__main__':

        #Start the cv bridge for imaging
	bridge = cv_bridge.CvBridge()

        #Initialize the ROS node
	print "starting node"
	rospy.init_node("rsdk_ik_to_limbs")
	callback_var = callback() #set the callback variable for the node

	print "changing cameras"
	try:
		cam_kill = CameraController("head_camera") #temporary holder to deactivate the head camera
		cam_kill.close()                           # not sure why, but both together crashes
	except:
		pass

        #Turn on the left eye-in-hand camera
	print "activating left camera"
	cam = CameraController("left_hand_camera") 
	cam.resolution = (320, 200) #Set the resolution (can't change it reliably, but fails without setting)
	cam.fps = 2 #Slower means less IO lag
	cam.gain = -1 #No gain
	cam.window = (530,350) #Display window
	cam.open() #Start it up!

        #Assign a callback for a camera message
	cam_callback_var = cam_callback()

	#Process samples block
	#Grab the background sample image
	bkg_img = cv2.imread("/home/prometheus/catkin_ws/src/var_aut/scripts/background_base2")

        #Grab the sample target object image files
	samps = ['bar','can','green_box','marker','mustard','nicotine_bottle','red_box','wire_nuts']
	fil_str = "/home/prometheus/catkin_ws/src/var_aut/scripts/img_ref/"

	#Get object ID histograms
	hist_labeled = []
	for a in samps: #For each object
                #Load the image
		img_o = cv2.imread(fil_str+a+".jpg")

                #Calculate RGB histograms
		histb = cv2.calcHist([img_o],[0],None,[32],[0,256])
		histg = cv2.calcHist([img_o],[1],None,[32],[0,256])
		histr = cv2.calcHist([img_o],[2],None,[32],[0,256])

		#add in labeled object with hist vectors
		hist_labeled = hist_labeled + [(a,[histb,histg,histr])]

	'''
	#Process sample images to subtract background
	# optional for making new target histograms
	for a in samps:
		img_o = cv2.imread(fil_str+a+".jpg")
		img_d = np.abs(bkg_img - img_o)
		img_d[0:30,0:40,:] = 0.0
		img_d[0:30,-40:,:] = 0.0
		img_d = cv2.blur(img_d,(5,5))
		img_d_0 = (img_d[:,:,0]>70.0)*1.0
		img_d_1 = (img_d[:,:,1]>70.0)*1.0
		img_d_2 = (img_d[:,:,2]>70.0)*1.0
		img_q = 1.0*((img_d_0+img_d_1+img_d_2)>0.5)
		img_o[:,:,0] = img_o[:,:,0]*img_q
		img_o[:,:,1] = img_o[:,:,1]*img_q
		img_o[:,:,2] = img_o[:,:,2]*img_q
		cv2.imwrite(fil_str+a+".jpg",img_o)
	'''

	print "Setting subscriptions"

        #Set camera callback
	cam_sub = rospy.Subscriber("/cameras/left_hand_camera/image", Image,cam_callback_var.camera_callback)

        #Subscribe to the arm coordinates and the in-arm IR sensor state
	rospy.Subscriber("/coords", String, callback_var.cb)

	ir_callback_var = ir_callback()
	rospy.Subscriber("/robot/range/left_hand_range/state",Range,ir_callback_var.ir_cb)

        #Start arm controller
	print "initializing arm"
	arm_init_ctr = 0 #counter to retry arm init- it's finnicky
	while arm_init_ctr < 5:
		try: #Try to start the arm
			print "Arms try: ",arm_init_ctr
			#Get the left arm interface
			left = baxter_interface.Limb('left')

			#activate the gripper
			grip_left = baxter_interface.Gripper('left', baxter_interface.CHECK_VERSION)
			if grip_left.error(): #reset the gripper on an error
				grip_left.reset()
			if (not grip_left.calibrated() and grip_left.type() != 'custom'):
                                #Calibrate if using the rtandard gripper and no error
				grip_left.calibrate()
			arm_init_ctr = 6 #end loop on success
		except: #count up attempt and wait a sec to try again
			arm_init_ctr+=1
			time.sleep(1.0)

	#table safety clear initialization
	#       limits for motions to clear the table contents
	aS_0 = [0.1,0.7,-0.4, 0.0,-3.142,0.0] 
	aS_1 = [0.1,0.7,0.2, 0.0,-3.142,0.0]
	aS_2 = [0.7,0.4,0.4, 0.0,-3.142,0.0]
	closeout_poses = [aS_0,aS_1,aS_2] #Doing them in sequence moves way up, over, then back

	print "Workspace clearance movements"
	grip_left.open() #Open gripper
	for i in [1,2]: #Do a quick, 'safe' motion
		print "move to position: "+ str(i)
		pos = closeout_poses[i] #Get goal
		pxr = pos[0] #Gext XYZ position
		pyr = pos[1]
		pzr = pos[2]
		owr,oxr,oyr,ozr = rpy_to_wxyz(pos[3],pos[4],pos[5]) #Make RPY of target to quaternion
		r_joints = get_IK('left',pxr,pyr,pzr,oxr,oyr,ozr,owr) #Solve IK 
		cam_sub.unregister() #Turn off cam- just have to
		left.move_to_joint_positions(r_joints) #Send motion command
		#Get camera back online
		cam_sub = rospy.Subscriber("/cameras/left_hand_camera/image", Image,
									cam_callback_var.camera_callback)

	# Initial work space pose
	pxr_o = 0.7
	pxr = pxr_o

	pyr_o = 0.4
	pyr = pyr_o

	pzr_o = 0.4
	pzr = pzr_o


	#Positional calibration data
	Z_MIN = -0.23
	Z_MAX_IR = -0.1 #reads 0.303 @ Z=-0.1 over table; 0.314 @ Z=-0.08

	#Vertical down: yaw controls rotation of gripper
	#Left-facing, standard orientation
	RPY_canon = [[0.0,-1.0*math.pi,0.0],[0.5*math.pi,0.0,0.0]] 

        #Simple init orientation
	r = -0.0*math.pi
	p = -1.0*math.pi
	y = 0.0*math.pi
	owr,oxr,oyr,ozr = rpy_to_wxyz(r,p,y)
	pos_p = [-1,-1,-1,-1,-1,-2] #Previous position no-read values
	r_grip_state = 0 #Gripper not closed

	#get JS
	js1 = joystick.joystick()
	js1.start()

        #tick counter
	n = 0

        #State machine state values
	START = 0
	PRINCIPLE_IMAGING = 1
	MOVE_ARM = 2
	LOCK_OUT = 3
	HOME_IN = 4
	CLOSE_IN = 5
	GRASP = 6
	IDLE = 7
	CLOSE_IN_2 = 8
	DROP = 9
	GET = 10
	JOYSTICK = 11
	JOYSTICK_XY = 12
	JOYSTICK_YAW = 13

	#Primary loop variables
	master_state = START
	RUNNING = True

        #COmmand history and object to grab
	cmd_prev = ['']
	object_sel = None

        #Autonomy level select
	# [1-5] Scrambled levels		Scale lvl
	# 1 User does angle only		3
	# 2 User doex XY and angle		6
	# 3 User does XY only			2
	# 4 Full auto				1
	# 5 User does ID only			4
	# 6 User does ID and angle		5
	AUTONOMY_LEVEL = 4

        #Run until killed
	print "starting loop"
	while RUNNING:

                #On first loop
		if n != 0:
                        #If awaiting a command already
			if (callback_var.cb_list != cmd_prev):
				cmd = callback_var.cb_list #Load them in
				cmd_prev = cmd
			else: #Otherwise, nothing
				cmd = ['none']
		else: #On other loops, command is home position
			cmd = ['home']
		n+=1 #Increment tick counter

                #If nothing to do, pass
		if (-1 in cmd):
			pass
		#If a motion command, and in a state to issue such
		elif (len(cmd) == 6) and (master_state in [IDLE,PRINCIPLE_IMAGING,CLOSE_IN,CLOSE_IN_2]):
			pos = cmd+[] #Load pos as command directly
			#if sufficiently different of a position
			if sum([abs(pos[i]-pos_p[i]) for i in range(6)])>0.001:
                                #Calculate target pose
				pxr = pos[0]
				pyr = pos[1]
				pzr = pos[2]
				roll = pos[3]
				pitch = pos[4]
				yaw = pos[5]
				owr,oxr,oyr,ozr = rpy_to_wxyz(pos[3],pos[4],pos[5])
				pos_p = pos+[] #update position history
				master_state = MOVE_ARM #Go to move state
			elif n != 0: #if not different enough and not first time around
				pos = pos + [-1] #Mark no position

                        #If not a no position
			if not(-1 in pos):
                                #Solve the IK
				r_joints = get_IK('left',pxr,pyr,pzr,oxr,oyr,ozr,owr)
			else:
                                #Otherwise, no joint poses to send
				r_joints = False

                #If a grasp command in a state it can execute
		elif (cmd[0].split("_")[0] == "G") and (master_state in [IDLE,PRINCIPLE_IMAGING,DROP]):
			master_state = GRASP #Send to grasp state

                #If a go home command, go to the homing state
		elif (cmd[0] == 'home'):
			master_state = HOME_IN

                #If stop command, go to the lockout state
		elif (cmd[0] == 'stop'):
			master_state = LOCK_OUT

                #If in the centering state, go to the close in 
		elif (cmd[0] == 'center'):
			master_state = CLOSE_IN

                #If a drop arm command, go to drop state
		elif (cmd[0] == 'drop'):
			master_state = DROP

                #If a get command, set state and note target object
		elif (cmd[0] == 'get'):
			master_state = GET
			object_sel = cmd[1]

                #If an autonomy level command
		elif (cmd[0] == 'aut'):
			master_state = HOME_IN #Go to home in state
			try: #Try to grab the autonomy level
				lvl = int(cmd[1])
				if 0<=lvl<=6: #If valid, set level
					AUTONOMY_LEVEL = int(cmd[1])
				else: #otherwise, invalid level
					print "Invalid Level"
			except: #If can't parse, note that
				print "Invalid entry"

                #if a joystick command, go to that state
		elif (cmd[0] == 'js'):
			master_state = JOYSTICK

		###
		#Actions block
		###

                # For the lockout state
		if (master_state == LOCK_OUT):
			print "Stopping System"
			cam_sub.unregister() #Deactivate the camera
			js1.stop() #Stop the joystick
			for i in [2,1,0]: #Run through the lockout poses
				pos = closeout_poses[i] #Get pose
				pxr = pos[0]
				pyr = pos[1]
				pzr = pos[2]
				owr,oxr,oyr,ozr = rpy_to_wxyz(pos[3],pos[4],pos[5])
				r_joints = get_IK('left',pxr,pyr,pzr,oxr,oyr,ozr,owr)
				left.move_to_joint_positions(r_joints) #execute pose
				time.sleep(0.3) #Wait a bit between motions
			cam.close() #Close the camers
			RUNNING = False #Stop running
			break #End loop 

                #For a grab command
		elif (master_state == GET):
			print "Getting: ",object_sel

                        #Check auto level 
			if AUTONOMY_LEVEL in [5,6]: #If 5 or 6, go to joystick
				master_state = JOYSTICK
			else: #Otherwise
				for a in range(len(locs)): #for each object label
					if obj_ids[a][0] == object_sel: #check if the selected object
						pxr = locs[a][0] #grab XY loc
						pyr = locs[a][1]
						pzr = -0.15 #z height to 15cm over
						owr,oxr,oyr,ozr = rpy_to_wxyz(0.0,-3.142,thetas[a][1]) #turn gripper to id'd angle
						yaw = thetas[a][1] 
						if AUTONOMY_LEVEL < 2: #if sufficiently autonomous
							owr,oxr,oyr,ozr = rpy_to_wxyz(0.0,-3.142,0.0) #Set angle
							theta = 0.0
							master_state = MOVE_ARM #Go to motion
							object_sel = "manual" #Flag manual execution
						elif AUTONOMY_LEVEL < 4: #If lower than four
							master_state = JOYSTICK #Goto js mode
							if AUTONOMY_LEVEL < 3: #less than three, set the angle auto
								owr,oxr,oyr,ozr = rpy_to_wxyz(0.0,-3.142,0.0)
						else: #Otherwise, lvl 4
							master_state = MOVE_ARM #goto arm motion
						r_joints = get_IK('left',pxr,pyr,pzr,oxr,oyr,ozr,owr) #Solve whatever IK we need
						print "Object at: ",locs[a][0],"x",locs[a][1],", ",thetas[a][1]," rad"
				if not(master_state in [MOVE_ARM,JOYSTICK]): #If not transitioned, object isn't there
					print "Object not found"
					object_sel = None
					master_state = PRINCIPLE_IMAGING #Go back to observing

                #In the idle state
		elif (master_state == IDLE):
			img_o = bridge.imgmsg_to_cv2(cam_callback_var.cb_img,"bgr8") #get an image update
			cv2.imshow("img_fetch",img_o) #Show image
			cv2.waitKey(1)

                #If in the closing-in phase
		elif (master_state == CLOSE_IN):
			print "Fine position adjust..."
			d_ct = 0 #reset counter
			com_x_avg = 0.0 #Average pose check
			img_ct = 0 #image counter
			master_state = CLOSE_IN_2 #Second half of close-in phase
			time.sleep(2.1) #cam update delay
			print "cam adj delay over"

                #Second half of close-in phase
		elif (master_state == CLOSE_IN_2):

                        #Open the gripper
			grip_left.open()

			#if an image available
			if (cam_callback_var.cb_img != -1):
                                #Get the image
				img_o = bridge.imgmsg_to_cv2(cam_callback_var.cb_img,"bgr8")
				img_o_cp = img_o+0 #copy of the image

                                #Subtract the backgroung
				img_d = np.abs(bkg_img - img_o)
				img_d[:40,0:100,:] = 0.0 #Block out region around gripper
				img_d[:40,-100:,:] = 0.0
				img_d[:,0:40,:] = 0.0
				img_d[:,-40:,:] = 0.0
				img_d[75:,:,:] = 0.0

                                #SMooth imahes, threshold RBG, and theshold sum
				img_d = cv2.blur(img_d,(4,4))
				img_d_0 = (img_d[:,:,0]>60.0)*(img_d[:,:,0]<120.0)*1.0
				img_d_1 = (img_d[:,:,1]>60.0)*(img_d[:,:,0]<120.0)*1.0
				img_d_2 = (img_d[:,:,2]>60.0)*(img_d[:,:,0]<120.0)*1.0
				img_q = 1.0*((img_d_0+img_d_1+img_d_2)>0.5)

                                #Filter out gripper parts
				img_o[:,:,0] = img_o[:,:,0]*img_d_0
				img_o[:,:,1] = img_o[:,:,1]*img_d_0
				img_o[:,:,2] = img_o[:,:,2]*img_d_0

                                #Mesh out pixel coords
				x = np.linspace(0, 319, 320)
				y = np.linspace(0, 199, 200)
				xv, yv = np.meshgrid(x, y)

                                #mask region pixels against coords
				x_set = img_q*xv
				y_set = img_q*yv

                                #Calculate area and COM
				area = np.sum(img_q)
				com_x = int(np.sum(x_set)/area)
				com_y = int(np.sum(y_set)/area)

                                #Update multiimage COM x position avarage
				com_x_avg = com_x_avg + com_x

                                #Output display
				img_o[com_y-2:com_y+2,com_x-2:com_x+2,0] = 0.0
				img_o[com_y-2:com_y+2,com_x-2:com_x+2,1] = 0.0
				img_o[com_y-2:com_y+2,com_x-2:com_x+2,2] = 255.0

                                #Lower and upper bounds
				cnt_l_b = 150
				cnt_u_b = 170
				img_ct+=1 #image counter
				if img_ct == 2 and d_ct < 5: #if second image count and under 5 image counts
					img_ct = 0 #Reset image counter
					dy = 0.0 #offsets
					dx = 0.0

					#If average COM less than the lower bound, horizontal motion
					if (com_x_avg/2.0 < cnt_l_b):
						if -0.75<yaw<0.75: #very small offset
							dy = 0.01
						else: #calculate yaw-based horizontal shift
							dx = 0.01*(yaw < -0.76) - 0.01*(yaw > -0.76)
					elif (com_x_avg/2.0 > cnt_u_b): #If greater than upper, verical motion
						if -0.75<yaw<0.75: #small adjustment
							dy = -0.01
						else: #calculate shift
							dx = -0.01*(yaw < -0.76) + 0.01*(yaw > -0.76)

                                        #If there's an adjustment to make
					if dy != 0.0 or dx != 0.0:
						d_ct+=1 #adjustment counter increment
						print "Adjustment ",d_ct
						pyr = pyr + dy #Make updates to position
						pxr = pxr + dx
						print "coords: ",pxr,pyr,pzr
						r_joints = get_IK('left',pxr,pyr,pzr,oxr,oyr,ozr,owr) #Recalculate IK
						left.move_to_joint_positions(r_joints) #Make adjustment
					com_x_avg = 0.0 #Reset COM average

                                        #If max adjustments made, or on target
					if (d_ct == 5) or ((dy == 0.0)and(dx==0.0)):
						if object_sel != None: #if an object to get
							master_state = DROP #go to drop state
						else: #otherwise, do nothing
							pass

                                #scale output image and highlight central selection region
				img_o = img_o*2.0
				img_o[:,cnt_l_b:cnt_u_b,1] = img_o[:,cnt_l_b:cnt_u_b,1] + 128.0

                                #Show output image
				cv2.imshow("img_fetch",img_o)
				cv2.waitKey(1)

                #For drop to table state
		elif (master_state == DROP):
			print "Dropping..."
			r_joints = get_IK('left',pxr,pyr,-0.23,oxr,oyr,ozr,owr) #Get IK for 23cm down
			if (r_joints != False): # if joint positions available,
				left.move_to_joint_positions(r_joints) #goto pose
				time.sleep(0.5) #wait a bit
				grip_left.close() #close the gripper
				time.sleep(1.5) #wait a bit
				master_state = HOME_IN #go to home state
			else:
				print "Failed." #Otherwise, IK failed

                #If in goto home phase
		elif (master_state == HOME_IN):
			print "Returning to home pos"
			print "Autonomy: ",AUTONOMY_LEVEL
			pos = closeout_poses[2] #Grab lift position and make pose
			pxr = pos[0]
			pyr = pos[1]
			pzr = pos[2]
			roll = pos[3]
			pitch = pos[4]
			yaw = pos[5]
			owr,oxr,oyr,ozr = rpy_to_wxyz(pos[3],pos[4],pos[5]) #Get angle
			pos_p = pos+[] #Update position prior
			cam_sub.unregister() #turn off camera
			r_joints = get_IK('left',pxr,pyr,pzr,oxr,oyr,ozr,owr) #Get ik
			left.move_to_joint_positions(r_joints) #go to home pose

			#Restart camera
			cam_sub = rospy.Subscriber("/cameras/left_hand_camera/image", Image,
									cam_callback_var.camera_callback)
			master_state = PRINCIPLE_IMAGING #go to imaging phase
			object_sel = None #clear object selection
			time.sleep(0.3)

                #For grasping state
		elif (master_state == GRASP):
			print "GRASP",cmd[0]
			if cmd[0] == 'G_open': #If an open command
				grip_left.open() #open the gripper
			elif cmd[0] == 'G_close': #if a close
				grip_left.close() #close it
			master_state = IDLE #goto idle state either way

                #For joystick control
		elif (master_state == JOYSTICK):
			_pxr = 160 #set default positions
			_pyr = 100
			theta = 0.0

                        #Goto joystick XY control mode
			master_state = JOYSTICK_XY

                #If in joystick yaw control mode
		elif (master_state == JOYSTICK_YAW):

                        #If an image available
			if (cam_callback_var.cb_img != -1):
                                #Grab image
				img_o = bridge.imgmsg_to_cv2(cam_callback_var.cb_img,"bgr8")

                                #Calculate angle from JS rotation
				if -1.56<theta + 0.05*js1.js_state[3]<1.56:
					theta = theta + 0.05*js1.js_state[3]

				#Calculate rotation transform
				vi = [0.0,80.0]
				vr = [math.cos(theta)*vi[0]-math.sin(theta)*vi[1],
						math.sin(theta)*vi[0]+math.cos(theta)*vi[1]]

                                #Display com XY 
				com_x = 160
				com_y = 30

                                #Draw UI indicator line for angle select
				cv2.line(img_o,(int(com_x+vr[0]),int(com_y+vr[1])),(com_x,com_y),(0,255,0),2)

                                #Prep output image and show
				img_r = cv2.resize(img_o, (0,0), fx=1.6, fy=1.6) 
				cv2.imshow("img_fetch",img_r)
				cv2.waitKey(1)

                        #On JS trigger
			if js1.js_state[4] == 1.0:
				print "Angle Registered"
				owr,oxr,oyr,ozr = rpy_to_wxyz(0.0,-3.14,-1.0*theta) #calculate angle
				r_joints = get_IK('left',pxr,pyr,pzr,oxr,oyr,ozr,owr) #get IK
				object_sel = 'man_yaw' #set select flag
				master_state = MOVE_ARM #go to arm movement

                        #Brief wait for smoothness
			time.sleep(0.1)

                #For joystick XY control
		elif (master_state == JOYSTICK_XY):

                        #If image available
			if (cam_callback_var.cb_img != -1):
                                #Grab image
				img_o = bridge.imgmsg_to_cv2(cam_callback_var.cb_img,"bgr8")

                                #Calculate target XY position from joystick XY
				_pxr = _pxr + 2.0*js1.js_state[0]
				_pyr = _pyr  + 2.0*js1.js_state[1]

                                #Get points as ints
				py = int(_pyr)
				px = int(_pxr)

                                #Prep and show UI image
				img_o[py-2:py+2,px-2:px+2,0] = 0.0
				img_o[py-2:py+2,px-2:px+2,1] = 255.0
				img_o[py-2:py+2,px-2:px+2,2] = 0.0 
				img_r = cv2.resize(img_o, (0,0), fx=1.6, fy=1.6) 
				cv2.imshow("img_fetch",img_r)
				cv2.waitKey(1)			

                        #On JS trigger
			if js1.js_state[4] == 1.0:

				print "Point Registered"

                                #set XY ranges
				Y_MAX = 0.91
				Y_MIN = 0.49
				X_MAX = 0.73
				X_MIN = 0.08

                                #Calibration offsets for image mapping
				offs_x = 0.4-0.003 #0.72->0.7 ; 0.585->0.6 ; 0.403 -> 0.4
				offs_y = 0.7-0.009 #0.21->0.22 ; 0.629->0.6 ; 0.707 -> 0.7
				c_offy = 0.06
				c_offx = 0.05
				alpha = 0.83

                                # if autonomy level less than 5
				if AUTONOMY_LEVEL < 5:
                                        #Calculate target pose values
					pxr = alpha*(1.0-(_pyr)/100.0)*((Y_MAX-Y_MIN)/2.0)+offs_y-c_offy
					pyr = alpha*(1.0-(_pxr)/160.0)*((X_MAX-X_MIN)/2.0)+offs_x-c_offx
					pzr = -0.15
					r_joints = get_IK('left',pxr,pyr,pzr,oxr,oyr,ozr,owr) #Solve IK
					object_sel = 'manual' #set to manual selection mode
					master_state = MOVE_ARM #move the arm
				else: #Otherwise
                                        #Get target positions
					px = alpha*(1.0-(_pyr)/100.0)*((Y_MAX-Y_MIN)/2.0)+offs_y-c_offy
					py = alpha*(1.0-(_pxr)/160.0)*((X_MAX-X_MIN)/2.0)+offs_x-c_offx

                                        #Loop over labels and find nearest target object
					d_min = 9999999
					label = None
					for a in range(len(locs)): #For each target
						_dist = (locs[a][0]-px)**2 + (locs[a][1]-py)**2 #get distance from target point
						if _dist < d_min: #find least distance target and grab coordinates from it
							d_min = _dist
							pxr = locs[a][0]
							pyr = locs[a][1]
							label = obj_ids[a][0]
							yaw = thetas[a][1]
                                        #For AL 5
					if AUTONOMY_LEVEL == 5:
						pzr = -0.15 #set positions from selected item label
						owr,oxr,oyr,ozr = rpy_to_wxyz(0.0,-3.142,yaw)
						r_joints = get_IK('left',pxr,pyr,pzr,oxr,oyr,ozr,owr)
						object_sel = label
						master_state = MOVE_ARM
                                        #For AL 6
					elif AUTONOMY_LEVEL == 6:
						pzr = -0.15 #get positions
						yaw = 0.0
						owr,oxr,oyr,ozr = rpy_to_wxyz(0.0,-3.142,yaw)
						r_joints = get_IK('left',pxr,pyr,pzr,oxr,oyr,ozr,owr)
						object_sel = 'manual' #set to manual selection
						AUTONOMY_LEVEL = 1 #follow steps for AL 1 henceforth
						master_state = MOVE_ARM #go move the arm

			time.sleep(0.1) #smoothness delay

		# Joint movement block
		elif (master_state == MOVE_ARM):
			print "Moving to: ",pxr,',',pyr,',',pzr,' @ ',pos[5],' rad',pos
			#If a valid IK was found
			if r_joints != False:
				try: #Try to...
					cam_sub.unregister() #turn off camera
					left.move_to_joint_positions(r_joints) #move the arm to target pose
					#Restart camera
					cam_sub = rospy.Subscriber("/cameras/left_hand_camera/image", Image,cam_callback_var.camera_callback)
					if object_sel != None: #if an object to get
						if object_sel == 'manual': #If manual
							if AUTONOMY_LEVEL < 3: #For AL < 3
								master_state = JOYSTICK_YAW #go to yaw angle mode
							else: #Otherwise, use autonomous angle, skip to close-inphase
								master_state = CLOSE_IN
						elif object_sel == 'man_yaw': #If manual yaw selected
							master_state = CLOSE_IN #go to close in
						else: #otherwise, also go to close in
							master_state = CLOSE_IN
					else: #if no object to select, go idle
						master_state = IDLE
				except: #on failure, do nothing
					pass

		# Imaging block
		elif (master_state == PRINCIPLE_IMAGING):

                        #On JS trigger, open the gripper (to see)
			if js1.js_state[5] == 1.0:
				grip_left.open()

                        #On cancel button, go to lockout
			if js1.js_state[6] == 1.0:
				master_state = LOCK_OUT

                        #If an image available
			if (cam_callback_var.cb_img != -1):
                                #Grab the image and copy it
				img_o = bridge.imgmsg_to_cv2(cam_callback_var.cb_img,"bgr8")
				img_o_cp = img_o.copy()

                                #Save sample images on occasion
				if cam_callback_var.ct > 30 and cam_callback_var.ct < 40:
					print "saving"
					cv2.imwrite("/home/prometheus/catkin_ws/src/var_aut/scripts/space_sample_"+str(cam_callback_var.ct)+".jpg",img_o_cp)

                                #Segment the image and get targets, labels, and locations
				img_o,thetas,labels,obj_ids,locs = img_segment_2(bkg_img, img_o,hist_labeled)

				n+=1 #tick counter

                                #Output image processing and display
				img_o_cp[98:102,158:162,0] = 0.0
				img_o_cp[98:102,158:162,1] = 0.0
				img_o_cp[98:102,158:162,2] = 255.0 
				img_o = np.concatenate((img_o,img_o_cp/255.0),axis=1)
				img_r = cv2.resize(img_o, (0,0), fx=1.6, fy=1.6) 
				cv2.imshow("img_fetch",img_r)
				cv2.waitKey(1)

                        # Otherwise, nothing to do
			else:
				print "No imaging"

		else: #short delay for alternative state (shouldn't ever get here)
			time.sleep(0.1)






