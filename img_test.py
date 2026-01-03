###
#
# Baxter robot materials handling and mixed-autonomy inverse
#  kinematics test. This software tests the Baxter image processing
#  system for pose estimation and tracking of objects from
#  the hand-in-eye camera system
#
###

#Standards
import math,time,random

#file handling
import glob

#image processing
import cv2

#For numeric and image processing
import numpy as np
from scipy import ndimage

#RGB section images
clr_blue = np.zeros((200,320,3))
clr_green = np.zeros((200,320,3))
clr_red = np.zeros((200,320,3))

#CPY section images
clr_cyan = np.zeros((200,320,3))
clr_purple = np.zeros((200,320,3))
clr_yellow = np.zeros((200,320,3))

#Light/dark images
clr_white = np.zeros((200,320,3))
clr_black = np.zeros((200,320,3))

#list of color arrays
clr_arrays = [clr_blue,clr_green,clr_red,
              clr_cyan,clr_purple,clr_yellow]

#Color array base colors
clr_vals = [(1.0,0.0,0.0),(0.0,1.0,0.0),(0.0,0.0,1.0),
            (1.0,1.0,0.0),(1.0,0.0,1.0),(0.0,1.0,1.0),]

#load base colors into color arrays
for i in range(len(clr_arrays)):
    clr_ar = clr_arrays[i]
    clr_ar[:,:,0] = np.ones((200,320))*clr_vals[i][0]
    clr_ar[:,:,1] = np.ones((200,320))*clr_vals[i][1]
    clr_ar[:,:,2] = np.ones((200,320))*clr_vals[i][2]
    clr_arrays[i] = clr_ar

#Image segmentation algorithm- revision 2
def img_segment_2(bkg_img, img_o,hist_labeled):

    #Grab a copy of input image
    img_o_i = img_o.copy() #cv2.blur(img_o,(3,3)) #Optional blur input option

    # Gamma adjusted color sum contouring
    img_int = np.sum(img_o_i,axis=2) #sum color channels
    img_int = img_int/(1.0*np.max(img_int)) #scale images
    for i in range(10): #iterative blur operation
        img_int = cv2.blur(img_int,(29-2*i,29-2*i))

    #Gamma-contrast adjusted array
    gamma = np.zeros(np.shape(img_o))
    gamma[:,:,0] = img_int
    gamma[:,:,1] = img_int
    gamma[:,:,2] = img_int

    #Apply gamma transform
    img_o_G = 255.0*((img_o_i/255.0)**(1.0*gamma))

    #smooth the gamma image
    img_o_i = cv2.blur(1.0*img_o_G,(3,3))
    img_o_i = np.uint8(img_o_i)

    #Grab RGB channels
    img_o_b = img_o_i[:,:,0]
    img_o_g = img_o_i[:,:,1]
    img_o_r = img_o_i[:,:,2]

    #grab and smooth a baseline intensity image
    img_o_i = np.sum(1.0*img_o.copy(),axis=2)
    img_o_i = cv2.blur(1.0*img_o_i,(5,5))
    img_o_i = np.uint8(img_o_i)

    #Grab edges from RGB and intensity images
    cont_i_i = cv2.Canny(img_o_i,30,60)
    cont_b_i = cv2.Canny(img_o_b,30,60)
    cont_g_i = cv2.Canny(img_o_g,30,60)
    cont_r_i = cv2.Canny(img_o_r,30,60)

    #find segment areas from color edges
    cont_seg = 1.0*((cont_b_i+cont_g_i+cont_r_i)>0)
    cont_seg = 1.0*(cv2.blur(1.0*cont_seg,(2,2))>0.20) #threshold segmentation over 20% level
    cont_seg_o = cont_seg.copy() #grab a copy

    #Color Partition
    img_clr = 1.0*img_o.copy() #copy of original image
    img_clr = cv2.blur(img_clr,(3,3)) #smoothing blur
    img_clr = img_clr/np.max(img_clr) #scale image
    img_clr_sum = np.sum(img_clr,axis=2) #intensity sum

    #scaled color channels
    img_clr[:,:,0] = img_clr[:,:,0]/img_clr_sum
    img_clr[:,:,1] = img_clr[:,:,1]/img_clr_sum
    img_clr[:,:,2] = img_clr[:,:,2]/img_clr_sum

    #smoothed color image
    img_clr = cv2.blur(img_clr,(5,5))

    #calculate average channel level
    img_clr_cor = np.zeros(np.shape(img_clr))
    clr_cor_avg = 1.0*np.average(img_clr,axis=2)

    #Calculate difference from color average
    img_clr_cor[:,:,0] = img_clr[:,:,0] - clr_cor_avg
    img_clr_cor[:,:,1] = img_clr[:,:,1] - clr_cor_avg
    img_clr_cor[:,:,2] = img_clr[:,:,2] - clr_cor_avg

    #black and white levels copy
    img_cor_bw = img_clr_cor.copy()

    #Threshold average difference image and scale
    img_clr_cor = img_clr_cor*(img_clr_cor>0.0)
    img_clr_cor = img_clr_cor/np.max(img_clr_cor)

    #threshold color intensity by 20% of the local sum level, then scale
    img_clr_sum = np.sum(img_clr_cor,axis=2)
    img_clr_sum = cv2.blur(img_clr_sum,(3,3))
    img_clr_sum = img_clr_sum*(img_clr_sum>0.2)
    img_clr_sum = img_clr_sum/(np.max(img_clr_sum))

    #scale thresholded color sum image to uint8 and find edges
    img_clr_sum = np.uint8(255.0*img_clr_sum)
    cont_clr_sum = cv2.Canny(img_clr_sum,30,60)

    #white & black check
    wb_ck = np.average(np.abs(img_cor_bw),axis=2)
    wb_ck = wb_ck/(np.max(wb_ck))

    #Gamma transform on w/b check
    gamma = 0.9
    wb_ck = (img_o_i/255.0)**(1.0*gamma) #apply gamma transform
    wb_ck = wb_ck/np.max(wb_ck)
    for i in range(5): #repeated gamma transform
        wb_ck = (wb_ck)**(1.0*gamma)
        wb_ck = wb_ck/np.max(wb_ck) #scale
    wb_ck = 1.0-wb_ck #Invert image
    wb_ck = np.uint8(255.0*wb_ck) #convert to uint8
    wb_cont = cv2.Canny(wb_ck,10,10) #find edges

    #Shadow handling segment
    img_o_i = np.sum(1.0*img_o.copy(),axis=2) #get intensity sum
    img_o_i = np.uint8(img_o_i) #convert to uint8

    #Build a standard uniform kernel
    kernel = np.ones((7,7))/49.0
    loc_avg = cv2.filter2D(img_o_i,-1,kernel) #apply kernel filter
    img_o_da = 1.0*img_o_i - loc_avg #subtract out filtered area
    img_o_k = img_o_i + 0.5*(img_o_da) #re-add half of subtracted image
    img_o_da = img_o_k - np.min(img_o_k) #remove minimum range
    img_o_da = img_o_da/np.max(img_o_da) #scale to 1.0

    #Apply level threshold
    img_o_da = 1.0*(img_o_da < 0.33)

    img_o_reg = 1.0*img_o.copy() #copy and scale regions image
    img_o_reg = img_o_reg/np.max(img_o_reg)
    img_o_reg = cv2.blur(img_o_reg,(2,2)) #smooth regions

    #make a small uniform kernel
    kernel = np.ones((5,5))
    kernel = kernel/(np.sum(kernel))

    #apply kernel filter to regions image by colot channel
    img_reg = np.zeros(np.shape(img_o_reg))
    img_reg[:,:,0] = cv2.filter2D(img_o_reg[:,:,0],-1,kernel)
    img_reg[:,:,1] = cv2.filter2D(img_o_reg[:,:,1],-1,kernel)
    img_reg[:,:,2] = cv2.filter2D(img_o_reg[:,:,2],-1,kernel)

    #find difference between original and filtered regions
    img_o_diff = img_o_reg - img_reg

    #re-level against minimum value and take magnitude
    img_o_diff1 = img_o_diff - np.min(img_o_diff)
    img_o_diff2 = np.abs(img_o_diff)

    #take magnitude of two difference levels and scale
    img_o_diff = img_o_diff2-img_o_diff1
    img_o_diff = np.abs(img_o_diff)
    img_o_diff = img_o_diff/np.max(img_o_diff)
    img_o_diff = 1.0 - img_o_diff #invert image

    #take mid-region between 5% and 25% of full scale, then re-level
    img_o_diff = img_o_diff*(img_o_diff > 0.05)*(img_o_diff < 0.25)
    img_o_diff = img_o_diff/np.max(img_o_diff)

    #SMooth second difference image and threshold over 20% limit
    img_o_diff2 = cv2.blur(img_o_diff,(7,7))
    img_o_diff2 = img_o_diff2*(img_o_diff2>0.2)

    #Calculate intensity sum and level
    img_o_diff2 = np.sum(img_o_diff2,axis=2)
    img_o_diff2 = img_o_diff2/np.max(img_o_diff2)

    #Threshold intensity sum to 40% level
    img_o_diff = np.sum(img_o_diff,axis=2)
    img_o_diff = img_o_diff*(img_o_diff2 > 0.4)

    #Output block
    img_o_i = np.zeros(np.shape(img_o)) #empty image
    img_o_i[:,:,0] = 255.0*img_o_diff #difference output
    img_o_i[:,:,1] = 255.0*img_o_da #regional segment output

    #holders for output lists (placeholders for test)
    thetas = []
    labels = []
    obj_ids = []
    locs = []

    #Return output image and holder lists
    return img_o_i/255.0,thetas,labels,obj_ids,locs

#Start up, loading segment test images
print "go"
fils = glob.glob("seg*.jpg")
print fils #print found files

#Image queue and counter
img_que = []
tivk = 0

#Looping over 4 tests each
for b in range(4):
    #for each test image
    for a in fils:
        img = cv2.imread(a) #Load image
        img_o,th,lab,obj,loc = img_segment_2(None,img,[]) #run segmentation

        #Concatenate output and original images for side-by-side
        img_o = np.concatenate((img_o,img/255.0),axis=1)
        img_r = cv2.resize(img_o, (0,0), fx=1.6, fy=1.6) #resize to display

        #Show wach image for 1/2 second
        cv2.imshow("img_fetch",img_r)
        cv2.waitKey(500)

#Note completion
print "done"
cv2.imshow("img_fetch",img_r) #show final image
cv2.waitKey(0) #await close

#Close image windows
cv2.destroyAllWindows()


