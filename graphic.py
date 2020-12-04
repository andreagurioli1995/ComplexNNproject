from PyQt5.QtWidgets import QMainWindow, QApplication ,QLabel, QMenu, QFrame, QMenuBar,QHBoxLayout,QVBoxLayout,QAction, QFileDialog, QWidget, QPushButton
from PyQt5.QtGui import QIcon, QImage, QPainter, QPen, QBrush,QPixmap
from PyQt5.QtCore import Qt, QPoint
import config 
import numpy as np
import sys
import pickle
from nn import nn

nn1=None
class prova(QLabel):
    
    def __init__(self):
        super().__init__()
        top = 400
        left = 400
        width = 400
        height = 400
        vettore=[]
        self.setStyleSheet("margin:5px; border:1px solid rgb(0, 255, 0); ")
        self.setGeometry(top, left, width, height)
        self.image = QImage(self.size(), QImage.Format_Grayscale8)
        
        self.image.fill(Qt.black)
        self.image
        self.drawing = False
        self.brushSize = 25
        self.brushColor = Qt.white
        self.brushColor2 = Qt.black
        self.lastPoint = QPoint()    
        self.setStyleSheet("border: 10px solid black;")
        self.mnist=[]
        self.test=[]  
        config.x = []
       
        
 
    def get_image(self):
        return self.image

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.lastPoint = event.pos()
            print(self.lastPoint)
       


    def mouseMoveEvent(self, event):
        if(event.buttons() & Qt.LeftButton) & self.drawing:
            painter = QPainter(self.image)
            painter.setPen(QPen(self.brushColor, self.brushSize, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            painter.drawLine(self.lastPoint, event.pos())
            self.lastPoint = event.pos()
            self.update()
        
        
    def mouseReleaseEvent(self, event):
         if event.button() == Qt.LeftButton:
            self.drawing = False
            self.update()
            self.saveImage()



    def paintEvent(self, event):
        canvasPainter  = QPainter(self)
        canvasPainter.drawImage(self.rect(),self.image, self.image.rect() )
        self.update()

    def saveImage(self):
        print(config.y)
        del self.mnist[:]
        del config.x [:]
        self.img_temp=self.image
        self.img = QImage((self.img_temp)).scaled(28, 28, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        print(self.img.isGrayscale())
        
        self.cont=0
        for i in range(28):
            for j in range(28):
                self.value=self.img.pixel(j,i)%256
                self.mnist.append(self.value)
        #Example().set_mnist(self.mnist)
        #print(self.mnist)
        self.update()
        if config.y==1:
            config.x=self.mnist
            print(config.x)
            self.image.fill(Qt.black)
            config.y=1
            config.x=[normalize(float(x)) for x in config.x]
            print(config.x)
            print(result(nn1.feedforward(config.x)))
            printNumber(config.x)
        #self.cancelImage()
        

    def cancelImage(self):
        #self.image = QImage(self.size(), QImage.Format_Grayscale8)
        self.image= None
        self.image = QImage(self.size(), QImage.Format_Grayscale8)
        self.image.fill(Qt.black)
        self.update()
     
    def get_mnist(self):
        #self.test.append(self.mnist)
        #print(self.test)
        return self.test

class Example(QWidget):
    
    def __init__(self):
        super().__init__()
        self.mnist_copy=[]
        self.initUI()
        config.y =1
    
    def set_mnist(self,a):
        self.mnist_copy=a

    def save(self):
        #prova().saveImage()

        print(result(nn1.feedforward(config.x)))
        printNumber(config.x)

        #print(config.x)
      
       
    
    def cancel(self):
       #prova().cancelImage()
       # self.image.fill(Qt.white)
       # self.update()
       config.y=1
       print(config.y)
       prova().saveImage()
       self.labelProva.clear()

    def initUI(self):
        
        layout1 = QHBoxLayout()
        layout2 = QVBoxLayout()
        layout3 = QVBoxLayout()

        labelImage = QLabel(self)

        self.labelProva = prova()
        labelImage.setStyleSheet("border: 1px solid black;") 
        pixmap = QPixmap("number.jpg")
        self.labelProva.setFixedSize(400, 400)
        self.labelProva.setStyleSheet("border: 100px solid black;") 
        labelImage.setPixmap(pixmap)
        labelImage.setFixedSize(400, 400)

        layout2.addWidget(self.labelProva)
        layout3.addWidget(labelImage)
        
       
        #okButton = QPushButton("OK")
        #cancelButton = QPushButton("Cancel")
        saveButton = QPushButton('Save', self)
        cancelButton = QPushButton('Cancel', self)
        saveAction = QAction(QIcon("icons/save.jpg"), "Save",self)
        saveAction.setShortcut("Ctrl+S")
        saveButton.addAction(saveAction)
        saveButton.clicked.connect(self.save)
      

        cancelAction = QAction(QIcon("icons/clear.png"), "Clear", self)
        cancelAction.setShortcut("Ctrl+C")
        cancelButton.addAction(cancelAction)
        cancelButton.clicked.connect(self.cancel)

        layout2.addWidget(saveButton)
        layout3.addWidget(cancelButton)
        #okButton.move(0, 0) 
        
        layout1.addLayout(layout2)
        layout1.addLayout(layout3)
        # moving the widget 
        # move(left, top) 
        

        #vbox.addLayout(label)
        widget = QWidget(self)
        widget.setLayout(layout1)
        self.setGeometry(0, 0, 825, 450)
        self.setWindowTitle('Buttons')
        self.show()

    


def main():
    
    app = QApplication(sys.argv)
    ex = Example()





 ######################################################################################
    #Reading
    global nn1
    nn1 = nn([28,16])
    numberOfEpochs = 10    # 10

    try:
        binary_file_pesi = open('Pesi.bin', mode='rb')   
        binary_file_bias = open('Bias.bin', mode='rb') 

        my_pesi=pickle.load(binary_file_pesi)
        nn1.setPesi(my_pesi)
        my_bias=pickle.load(binary_file_bias)
        nn1.setBias(my_bias)
        print("Pesi già presenti!")

    #training if reading fails  
    except IOError:
        print("Pesi mancanti, inizio training")
        ######################################################################################
        # input dal file


        targetsTrain = []
        inputsTrain = []
        #mydataset = open("data/mnistTrain.txt", "r")
        mydataset = open(
            r"C:\\Users\\bigfo\\OneDrive\\Desktop\\dati\\mnistTrain_copy.txt", "r")
        for x in range(30000):  # numberOfinputs 30000
            targetTrain = int(mydataset.read(1))
            number = [normalize(float(x)) for x in next(mydataset).split()]
            #number = [1 if int(x)>90 else 0 for x in next(mydataset).split()]
            targetsTrain.append(targetTrain)
            inputsTrain.append(number)
        # vettori di targetTrain
        targetVectors = []
        for i in range(len(targetsTrain)):
            targetVectors.append(
                [float(1) if x == targetsTrain[i] else float(0) for x in range(10)])

        mydataset.close()


        ######################################################################################
        #Testing data

        targetsTest = []
        inputsTest = []
        #testDataset = open("data/mnistTest.txt", "r")
        testDataset = open(
            r"C:\\Users\\bigfo\\OneDrive\\Desktop\\dati\\mnistTest_copy.txt", "r")
        for x in range(10000):     # len(inputsTest) == 10000
            targetTest = int(testDataset.read(1))
            numberTest = [normalize(float(x)) for x in next(testDataset).split()]
            targetsTest.append(targetTest)
            inputsTest.append(numberTest)

        # vettori di targetTrain
        targetsTestVectors = []
        for i in range(len(targetsTest)):
            targetsTestVectors.append(
                [float(1) if x == targetsTest[i] else float(0) for x in range(10)])

        testDataset.close()

        for l in range(numberOfEpochs):
            nn1.TrainNet(inputsTrain, targetVectors)
            print("EFFICIENCY", l+1, ": ", getEfficiency(inputsTest, targetsTest))

        tempPesi=pickle.dump(nn1.getPesi(),open('Pesi.bin', 'wb'))
        tempBias=pickle.dump(nn1.getBias(),open('Bias.bin', 'wb'))






    ######################################################################################
    sys.exit(app.exec_())



######################################################################################
# funzioni ausiliarie

def result(output) : 
    max=0
    count=0
    for z in range(len(output)):
        if output[z]>max:
            max=output[z]
            count=z
    return count        
    
def normalize(x) :
    grey = 90
    return 1.0/(1.0+np.exp(-x+grey))

def getEfficiency(inputsVector, targetScalars) :
    c = 3
    efficiency = 0.0
    for x in range(len(inputsVector)):
        output=nn1.feedforward(inputsVector[x])
        #print("test", x+1, "(targetTrain, ris): ", targetVector[x], result(output), "      ", targetsTrain[x] == result(output))
        if targetScalars[x] == result(output) :
            efficiency += 1.0
        elif 300*c < x < 300*(c+1) :  # elif temporaneo per vedere dove sbaglia
            pass
            #printNumber(inputsVector[x])
    efficiency = efficiency/len(inputsVector)
    return efficiency

def printNumber(n) :
    s = normalize(110)
    for i in range(28) :
        for j in range(28) :
            if(n[i * 28 + j] > s) :
                print("o", end =" ")
            else :
                print(" ", end =" ")
        print("")
    

















if __name__ == '__main__':
    main()


#print("prova") 




####################################################################################################################################

