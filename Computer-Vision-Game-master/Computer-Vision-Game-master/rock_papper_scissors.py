import numpy as np
import cv2
import keras
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import time
from sklearn.metrics import pairwise


model_path="rock_papper_scissors_model.h5"
model = keras.models.load_model(model_path)

background = None
accumulated_weight = 0.5

ROI_top_1 = 100
ROI_bottom_1 = 300
ROI_right_1 = 400
ROI_left_1 = 600

ROI_top_2 = 100
ROI_bottom_2 = 300
ROI_right_2 = 50
ROI_left_2 = 250

def winner_check(player_1_choice,player_2_choice):
    if(player_1_choice=="Rock"):
        if(player_2_choice=="Rock"):
            return 0
        elif(player_2_choice=="Scissors"):
            return 1
        elif(player_2_choice=="Paper"):
            return 2
    elif(player_1_choice=="Scissors"):
        if(player_2_choice=="Rock"):
            return 2
        elif(player_2_choice=="Scissors"):
            return 0
        elif(player_2_choice=="Paper"):
            return 1
    elif(player_1_choice=="Paper"):
        if(player_2_choice=="Rock"):
            return 1
        elif(player_2_choice=="Scissors"):
            return 2
        elif(player_2_choice=="Paper"):
            return 0
    else:
        return 3

def cal_accum_avg(frame, accumulated_weight):
    global background
    if background is None:
        background = frame.copy().astype("float")
        return None
    cv2.accumulateWeighted(frame, background, accumulated_weight)

def segment_hand(frame, threshold=25):
    global background
    diff = cv2.absdiff(background.astype("uint8"), frame)
    _ , thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    #Fetching contours in the frame (These contours can be of hand or any other object in foreground) ...
    contours, hierarchy = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # If length of contours list = 0, means we didn't get any contours...
    if len(contours) == 0:
        return None
    else:
        # The largest external contour should be the hand 
        hand_segment_max_cont = max(contours, key=cv2.contourArea)
        # Returning the hand segment(max contour) and the thresholded image of hand...
        return (thresholded, hand_segment_max_cont)

def count_fingers(thresholded, hand_segment):
    # Calculates the convex hull of the hand segment
    conv_hull = cv2.convexHull(hand_segment)
    # Now the convex hull will have at least 4 most outward points, on the top, bottom, left, and right.
    # Finds the top, bottom, left , and right.
    top    = tuple(conv_hull[conv_hull[:, :, 1].argmin()][0])
    bottom = tuple(conv_hull[conv_hull[:, :, 1].argmax()][0])
    left   = tuple(conv_hull[conv_hull[:, :, 0].argmin()][0])
    right  = tuple(conv_hull[conv_hull[:, :, 0].argmax()][0])
    # In theory, the center of the hand should be half way between the top and bottom and halfway between left and right
    cX = (left[0] + right[0]) // 2
    cY = (top[1] + bottom[1]) // 2
    # finds the maximum euclidean distance between the center of the palm
    # and the most extreme points of the convex hull
    # Calculates the Euclidean Distance between the center of the hand and the left, right, top, and bottom.
    distance = pairwise.euclidean_distances([(cX, cY)], Y=[left, right, top, bottom])[0]
    # Grab the largest distance
    max_distance = distance.max()
    # Create a circle with 80% radius of the max euclidean distance
    radius = int(0.8 * max_distance)
    circumference = (2 * np.pi * radius)
    # Not grab an ROI of only that circle
    circular_roi = np.zeros(thresholded.shape[:2], dtype="uint8")
    # draw the circular ROI
    cv2.circle(circular_roi, (cX, cY), radius, 255, 10)
    # Using bit-wise AND with the cirle ROI as a mask.
    # This then returns the cut out obtained using the mask on the thresholded hand image.
    circular_roi = cv2.bitwise_and(thresholded, thresholded, mask=circular_roi)
    # Grab contours in circle ROI
    contours, hierarchy = cv2.findContours(circular_roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # Finger count starts at 0
    count = 0
    # loop through the contours to see if we count any more fingers.
    for cnt in contours: 
        (x, y, w, h) = cv2.boundingRect(cnt)
        # Increment count of fingers based on two conditions:
        # 1. Contour region is not the very bottom of hand area (the wrist)
        out_of_wrist = ((cY + (cY * 0.25)) > (y + h))
        # 2. Number of points along the contour does not exceed 25% of the circumference of the circular ROI (otherwise it's counting points off the hand)
        limit_points = ((circumference * 0.25) > cnt.shape[0])
        if  out_of_wrist and limit_points:
            count += 1
    return count

cam = cv2.VideoCapture(0)
num_frames =0
word_dict = {0:'Rock',1:'Rock',2:'Scissors',3:'Scissors',4:'Paper',5:'Paper',6:"Invalid",7:"Invalid",8:"Invalid",9:"Invalid",10:"Invalid"}
flag=False

while True:
    start=22
    #games_counter=start-int(num_frames/10)
    games_counter=start-int(num_frames/1000000000)
    ret, frame = cam.read()

    # filpping the frame to prevent inverted image of captured frame...
    frame = cv2.flip(frame, 1)

    frame_copy = frame.copy()

    # ROI from the frame
    roi_1 = frame[ROI_top_1:ROI_bottom_1, ROI_right_1:ROI_left_1]
    roi_2 = frame[ROI_top_2:ROI_bottom_2, ROI_right_2:ROI_left_2]

    gray_frame_1 = cv2.cvtColor(roi_1, cv2.COLOR_BGR2GRAY)
    gray_frame_2 = cv2.cvtColor(roi_2, cv2.COLOR_BGR2GRAY)
    gray_frame_1 = cv2.GaussianBlur(gray_frame_1, (9, 9), 0)
    gray_frame_2 = cv2.GaussianBlur(gray_frame_2, (9, 9), 0)


    if num_frames < 60:
        
        cal_accum_avg(gray_frame_1, accumulated_weight)
        #cal_accum_avg(gray_frame_2, accumulated_weight)
        cv2.putText(frame_copy, "FETCHING BACKGROUND...PLEASE WAIT", (80, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
    
    else: 
        # segmenting the hand region
        hand_1 = segment_hand(gray_frame_1)
        hand_2 = segment_hand(gray_frame_2)
        

        # Checking if we are able to detect the hand...
        if (hand_1 is not None) and (hand_2 is not None):
            
            
            thresholded_1, hand_segment_1 = hand_1
            thresholded_2, hand_segment_2 = hand_2

            # Drawing contours around hand segment
            cv2.drawContours(frame_copy, [hand_segment_1 + (ROI_right_1, ROI_top_1)], -1, (255, 0, 0),1)
            cv2.drawContours(frame_copy, [hand_segment_2 + (ROI_right_2, ROI_top_2)], -1, (255, 0, 0),1)
            
            cv2.imshow("Thesholded Hand Image_1", thresholded_1)
            cv2.imshow("Thesholded Hand Image_2", thresholded_2)
            
            #thresholded_1 = cv2.resize(thresholded_1, (64, 64))
            #thresholded_2 = cv2.resize(thresholded_2, (64, 64))
            #thresholded_1 = cv2.cvtColor(thresholded_1, cv2.COLOR_GRAY2RGB)
            #thresholded_2 = cv2.cvtColor(thresholded_2, cv2.COLOR_GRAY2RGB)
            #thresholded_1 = np.reshape(thresholded_1, (1,thresholded_1.shape[0],thresholded_1.shape[1],3))
            #thresholded_2 = np.reshape(thresholded_2, (1,thresholded_2.shape[0],thresholded_2.shape[1],3))
            
            #pred_1 = model.predict(thresholded_1)
            pred_1 = count_fingers(thresholded_1,hand_segment_1)
            #pred_2 = model.predict(thresholded_2)
            pred_2 = count_fingers(thresholded_2,hand_segment_2)
            #cv2.putText(frame_copy, word_dict[np.argmax(pred_1)], (500, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            cv2.putText(frame_copy, word_dict[pred_1], (500, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2) 
            cv2.putText(frame_copy, word_dict[pred_2], (50, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            
            if games_counter<=100 and games_counter>0:
                cv2.putText(frame_copy, "Play!", (200, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                
            if(games_counter==100):
                winner=winner_check(word_dict[pred_2],word_dict[np.argmax(pred_1)])
                
                if winner==0:
                    cv2.putText(frame_copy, "It's a Tie", (200, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                    flag=True
                elif winner==3:
                    
                    cv2.putText(frame_copy, "Invalid Input", (200, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                    flag=True
                else:
                    cv2.putText(frame_copy, f"Player {winner} wins", (200, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                    flag=True
                    
    # Draw ROI on frame_copy
    cv2.rectangle(frame_copy, (ROI_left_1, ROI_top_1), (ROI_right_1, ROI_bottom_1), (255,128,0), 3)
    cv2.rectangle(frame_copy, (ROI_left_2, ROI_top_2), (ROI_right_2, ROI_bottom_2), (255,128,0), 3)
    # incrementing the number of frames for tracking
    num_frames += 1

    # Display the frame with segmented hand

    cv2.putText(frame_copy, "Player 2", (430, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    cv2.putText(frame_copy, "Player 1", (80, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    cv2.putText(frame_copy, f"Game clock {int(games_counter/2)}", (10, 450), cv2.FONT_ITALIC, 0.5, (51,255,51), 2)
    cv2.imshow("Sign Detection", frame_copy)


    # Close windows with Esc
    k = cv2.waitKey(1) & 0xFF

    if k == 27 or flag==True or games_counter<0:
        time.sleep(2)
        break

# Release the camera and destroy all the windows
cam.release()
cv2.destroyAllWindows()

