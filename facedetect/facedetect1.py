import cv2 # type: ignore

eid=101;name="Naman";dept="Accountant";

def drawBoundary(img,classifier,scalefactor,minNbr,text,clf):
    gray_image=cv2.cvtColor(img,cv2,COLOR_BGR2GRAY)
    features=classifier.detectMultiScale(gray_image,scalefactor,minNbr)
    cord=[]
    for(x,y,w,h) in features:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
        eid,predict=clf.predict(gray_image[y:y+h,x:x+w])
        confidence=int((100*(1-predict/300)))
        print(confidence)
        if(confidence>77):
            cv2.putText((img,"Employee_ID : {eid}"),
            (x,y-55),cv2.FONT_HERSHEY_COMPLEX,0.8,(255,255,255,3))
            cv2.putText((img,"Employee_Name : {name}"),
            (x,y-30),cv2.FONT_HERSHEY_COMPLEX,0.8,(255,255,255,3))
            cv2.putText((img,"Department : {dept}"),
            (x,y-5),cv2.FONT_HERSHEY_COMPLEX,0.8,(255,255,255,3))
        else:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
            cv2.putText((img,"Unknown Person"),(x,y-55),
            cv2.FONT_HERSHEY_COMPLEX,0.8,(255,255,255,3))

        cord=[x,y,w,h]
        return cord
    
def reconize(img,faceCascade,clf):
    cord=drawBoundary(img,faceCascade,1.1,10,(255,25,255),"Face",clf)
    return img

faceCascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
clf=cv2.face.LBPHFaceRecognizer()
print(faceCascade)
video_cap=cv2.VideoCapture(0)
while(True):
    ret,img=video_cap.read()
    cv2.imshow('video',img)
    k=cv2.waitKey(30)
    if(k==27):   # esc
        break
video_cap.release()
cv2.destroyWindow()