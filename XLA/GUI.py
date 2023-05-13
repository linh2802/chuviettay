from tkinter import *
import tkinter
from PIL import ImageTk, Image, ImageDraw
import PIL

import numpy as np
import cv2
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn import neighbors
from sklearn.metrics import accuracy_score

img = cv2.imread('digits.png', 0)

cells = [np.hsplit(row, 50) for row in np.vsplit(img, 50)]
#70% data for train 30% for test
trainVar = np.array(cells)[:,:35].reshape(-1, 400)
testVar = np.array(cells)[:,35:50].reshape(-1, 400)
#Labels
k = np.arange(10)
trainLabels = np.repeat(k, 175)
testLabels = np.repeat(k, 75)

countImages = 1
kVar = 40
clf = neighbors.KNeighborsClassifier(n_neighbors = kVar, p = 2, weights = 'distance')#p = 2 the standard Euclidean distance
def kNN():
    global trainVar, trainLabels, testVar, clf, kVar
    clf = neighbors.KNeighborsClassifier(n_neighbors = kVar, p = 2, weights = 'distance')#p = 2 the standard Euclidean distance
    clf.fit(trainVar, trainLabels)
    y_pred = clf.predict(testVar)
    lbl2['text'] = "%.2f %%" %(100*accuracy_score(testLabels, y_pred))

def PaintNums(event):
    color='white'
    x1, y1 = (event.x-7), (event.y-7)
    x2, y2 = (event.x+7), (event.y+7)
    cv.create_oval(x1, y1, x2, y2, fill=color, outline=color, width=5)
    draw.line([x1, y1, x2, y2], fill=color, width=5)

def CheckImage():
    global countImages, clf, image1, testVar
    filename = "Images/image" + str(countImages) + ".png"
    image1.save(filename)
    countImages += 1
    image = cv2.imread(filename, 0)
    image = cv2.resize(image, (20, 20))
    print(image)
    kNN()
    lbl1['text'] = str(clf.predict(image.reshape(-1, 400))[0])
    
def ClearImage():
    global image1, draw
    cv.delete("all")
    image1 = PIL.Image.new("RGB", (200, 200), "black")
    draw = ImageDraw.Draw(image1)

def SetPara():
    global kVar
    kVar = int(inputtxt1.get(1.0, END))
    print(kVar)

window = Tk()
window.title("Handwritten Digit Recognition")
window.geometry("530x450")
window.resizable(False, False)

#add background
canvas1 = Canvas(window, width=700, height=600, bg='white')
canvas1.place(x = 0, y = 0) 
canvas1.create_rectangle(0, 80, 700, 600, fill="#A5C9FF", outline="#A5C9FF")
canvas1.create_rectangle(33, 123, 250, 360, fill="white", outline="black")
canvas1.create_rectangle(304, 158, 504, 208, fill="white", outline="black")
canvas1.create_rectangle(33, 374, 250, 424, fill="white", outline="black")
canvas1.create_rectangle(325, 247, 379, 319, fill="white", outline="white")
#add logo
img1 = PIL.Image.open("logoBK.png")
img1 = img1.resize(((60, 60)))
photoImg = PIL.ImageTk.PhotoImage(img1)
lbl1 = tkinter.Label(window, image=photoImg, bg='white', width=80, anchor=W)
lbl1.place(x = 10, y = 10)
#add text1: Draw
lbl = tkinter.Label(window, text="Draw",bg='white', fg="black", font=("Segoe UI", 13), anchor=E)
lbl.place(x = 40, y = 128)
#add text: k
lblk = tkinter.Label(window, text="      k  ",bg='white', fg="black", font=("Segoe UI", 13), anchor=E)
lblk.place(x = 304, y = 247)
#add draw box
cv = Canvas(window, width=200, height=200, bg='black')
cv.place(x = 40, y = 153)
image1 = PIL.Image.new("RGB", (200, 200), "black")
draw = ImageDraw.Draw(image1)
cv.bind('<B1-Motion>', PaintNums)
#add button Chech
iB1 = PIL.Image.open("iconCheck.png")
photoImg1 = PIL.ImageTk.PhotoImage(iB1)
button1 = tkinter.Button(window, image=photoImg1, command=CheckImage, bg = 'white', width=60,height=60)
button1.place(x = 304, y = 360)
#add button Clear
iB2 = PIL.Image.open("iconClear.png")
photoImg2 = PIL.ImageTk.PhotoImage(iB2)
button2 = tkinter.Button(window, image=photoImg2, command=ClearImage, bg = 'white', width=60,height=60)
button2.place(x = 437, y = 360)
#add button Set
iB3 = PIL.Image.open("iconSet.png")
photoImg3 = PIL.ImageTk.PhotoImage(iB3)
button3 = tkinter.Button(window, image=photoImg3, command=SetPara, bg = 'white', width=60,height=60)
button3.place(x = 437, y = 250)
#add text2: Number
lbl1 = tkinter.Label(window, text="Number",bg='white', fg="black", font=("Segoe UI", 13), anchor=E)
lbl1.place(x = 135, y = 383)
#add text3: %
lbl2 = tkinter.Label(window, text="%",bg='white', fg="black", font=("Segoe UI", 17), anchor=E)
lbl2.place(x = 373, y = 165)
#input Text 1
inputtxt1 = Text(window, height = 2, width = 8,fg='white', bg = "#0035F2", font=("Segoe UI", 13))
inputtxt1.place(x = 304, y = 271)

window.mainloop()