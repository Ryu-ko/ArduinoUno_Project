import glob,cv2,random
import numpy as np
#from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input,decode_predictions
#from keras.preprocessing.image import ImageDataGenerator,load_img,img_to_array
from keras.models import load_model

model=load_model('class_model.h5')

camera=cv2.VideoCapture(0);
top, right, bottom, left = 10, 350, 225, 590

l=[]
for i in glob.glob('computer_images/*'):
	l.append(i)

i=0
count=0
old=2
user_score=0
computer_score=0
total_moves=0
user_tag="None"
computer_tag="None"
result="(None :|)"
print("\n\n")
print("| Move Count |"," Computer Score |"," User Score |")
while(True):
	i+=1
	grabbed,frame=camera.read()
	if(i%30==0):
		if(old==0):
			count=0;
			old=3
		old-=1
	
	keypress = cv2.waitKey(1) & 0xFF
	frame=cv2.flip(frame,1)
	cv2.rectangle(frame,(left,top),(right,bottom),(0,255,0),2)
	cv2.putText(frame,str(old),(0,100), cv2.FONT_HERSHEY_SIMPLEX, 4,(255,255,255),2,cv2.LINE_AA)
	cv2.putText(frame,str(user_tag),(0,400), cv2.FONT_HERSHEY_SIMPLEX, 2,(255,255,255),2,cv2.LINE_AA)
	cv2.putText(frame,str(result),(0,450), cv2.FONT_HERSHEY_SIMPLEX, 2,(255,255,255),2,cv2.LINE_AA)
	cv2.imshow("Video Feed", frame)
	if(old==0 and count==0):
		
		# FOR USER TAG
		user_move=frame[10:225,350:590]
		image1=np.expand_dims(user_move,axis=0)
		image1=preprocess_input(image1)
		yhat=model.predict(image1)
		temp=list(yhat[0])
		class_index=temp.index(np.amax(yhat))
		if(class_index==0):
			user_tag="Paper"
		elif(class_index==1):
			user_tag="Rock"
		else:
			user_tag="Scissors"
		# cv2.putText(frame,str(user_tag),(50,100), cv2.FONT_HERSHEY_SIMPLEX, 4,(255,255,255),2,cv2.LINE_AA)
		# FOR COMPUTER TAG
		var=random.choice(l)
		temp=var.split('/')[1][0]
		if(temp=='p'):
			computer_tag="Paper"
		elif(temp=='r'):
			computer_tag="Rock"
		else:
			computer_tag="Scissors"
		
		# FOR POINT TABLE
		if(user_tag=="Rock"):
			if(computer_tag=="Scissors"):
				result="You Won :)"
				user_score+=1
			elif(computer_tag=="Paper"):
				result="You Loose :("
				computer_score+=1
			else:
				result="Draw :/"
			total_moves+=1
		elif(user_tag=="Scissors"):
			if(computer_tag=="Paper"):
				result="You Won :)"
				user_score+=1
			elif(computer_tag=="Rock"):
				result="You Loose :("
				computer_score+=1
			else:
				result="Draw :/"
			total_moves+=1
		else:
			if(computer_tag=="Rock"):
				result="You Won :)"
				user_score+=1
			elif(computer_tag=="Scissors"):
				result="You Loose :("
				computer_score+=1
			else:
				result="Draw :/"
			total_moves+=1

		# PRINTING STATS
		print("|      ",total_moves,"             ",computer_score,"           ",user_score,"   |")
		cv2.imshow("computer",cv2.imread(var))
		count=1;

	# cv2.imshow("Video Feed", frame)
	if (keypress == ord("q")):
		break
camera.release()
cv2.destroyAllWindows()
# Mat merged = Mat(Size(imgFixedSize.width*2, imgFixedSize.height*2), CV_8UC3);
