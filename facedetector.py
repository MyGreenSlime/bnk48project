import numpy as np
import cv2
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import pickle
class facedetect:
    def __init__(self):
        self.subjects = ["", "CAN", "CHERPRANG","IZURINA","JAA",'JANE','JENNIS','JIB','KAEW','KAIMOOK','KATE','KORN','MAYSA','MIND','MIORI','MOBILE','MUSIC','NAMNUENG','NAMSAI','NINK',"NOEY","ORN","PIAM","PUN","PUPE","SATCHAN","TARWAAN"]
        self.model = pickle.load(open("LRmodel05.model", 'rb'))
        #self.model.read('./model21_1000.xml')
        self.face_cascade = cv2.CascadeClassifier('opencv-files/lbpcascade_frontalface.xml')
    def detect(self,img):
        pic = img.copy()
        gray_img = cv2.GaussianBlur(pic,(41,41),0) #v1 = 21,21
        gray_img = cv2.cvtColor(gray_img,cv2.COLOR_BGR2GRAY)
        facepre = cv2.cvtColor(pic,cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray_img,scaleFactor =1.05,minNeighbors = 5)
        labellist = []
        labeltextlist = []
        for x,y,w,h in faces:
            #label= self.model.predict(cv2.resize(facepre[y:y+w, x:x+h], (64,64), interpolation=cv2.INTER_CUBIC))
            resize = cv2.resize(facepre[y:y+w, x:x+h], (64,64), interpolation=cv2.INTER_CUBIC)
            flat = resize.flatten()
            label = self.model.predict(flat.reshape(1,-1))
            labellist.append(label)
            labeltextlist.append(self.subjects[label[0]])
        print(labeltextlist)
        count = 0
        for x,y,w,h in faces:
            cv2.rectangle(pic, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(pic, labeltextlist[count], (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
            count+=1
        return pic,labeltextlist
    def predict(self,path):
        self.img = cv2.imread(path)
        self.predicted_img1,labellist = self.detect(self.img)
        self.resize = cv2.resize(self.predicted_img1, (int(self.img.shape[1]),int(self.img.shape[0])), interpolation=cv2.INTER_CUBIC)
        return self.resize,labellist