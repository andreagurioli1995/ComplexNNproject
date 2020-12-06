from PyQt5.QtWidgets import QMainWindow, QApplication ,QLabel, QMenu, QFrame, QMenuBar,QHBoxLayout,QVBoxLayout,QAction, QFileDialog, QWidget, QPushButton
from PyQt5.QtGui import QIcon, QImage, QPainter, QPen, QBrush,QPixmap
from PyQt5.QtCore import Qt, QPoint
import config 
import numpy as np
import sys
import pickle
from random import randint
from nn import nn

class QLabelDraw(QLabel):
    
    def __init__(self):
        super().__init__()
        top = 400
        left = 400
        width = 400
        height = 400

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

    #funzioni di disegno tramite lo spostamento del mouse 
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.lastPoint = event.pos()

    #funzioni di disegno tramite lo spostamento del mouse    
    def mouseMoveEvent(self, event):
        if(event.buttons() & Qt.LeftButton) & self.drawing:
            painter = QPainter(self.image)
            painter.setPen(QPen(self.brushColor, self.brushSize, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            painter.drawLine(self.lastPoint, event.pos())
            self.lastPoint = event.pos()
            self.update()
    
    #funzioni di disegno tramite lo spostamento del mouse    
    def mouseReleaseEvent(self, event):
         if event.button() == Qt.LeftButton:
            self.drawing = False
            self.update()

    def paintEvent(self, event):
        canvasPainter  = QPainter(self)
        canvasPainter.drawImage(self.rect(),self.image, self.image.rect() )
        self.update()

    #salva l'immagine disegnata 
    def saveImage(self):
        del self.mnist[:]
        
        #resize dell'immagine da 400x400 a 28x28
        self.img = QImage((self.image)).scaled(28, 28, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)

        #salvataggio del colore dei pixel nell'array msist
        for i in range(28):
            for j in range(28):
                self.value=self.img.pixel(j,i)%256
                self.mnist.append(self.value)

        #inserimento del vettore mnist nella rete neurale
            #config.x=[normalize(float(x)) for x in config.x]
        #----self.mnist=[1 if int(x)>90 else 0 for x in self.mnist]
        #----print(result(nn1.feedforward(self.mnist)))
        #result=result(nn1.feedforward(self.mnist))
        print(self.mnist)
        result =randint(0, 9)
        return result
        
    #cancella l'immagine disegnata
    def cancelImage(self):
        self.image.fill(Qt.black)
        self.update()

class Example(QWidget):
    
    def __init__(self):
        super().__init__()
        self.mnist_copy=[]
        self.initUI()
    
    #richiama la funzione di salvataggio in QLabelDraw returnando il valore stampato dalla rete neurale
    def save(self):
        
        nn_result=self.QLabelDraw.saveImage()
        
        #inserisce l'immagine .jpg rappresentante il numero finale nel labelImage
        self.labelImage.clear()
        final_image='C:\\Users\\giovi\\Documents\\GitHub\\ComplexNNproject\\number\\number'+str(nn_result)+'.jpg'
        self.pixmap = QPixmap(final_image)
        self.labelImage.setPixmap(self.pixmap)
       #print(result(nn1.feedforward(config.x)))

    #richiama la funzione di cancellazione del disegno e cancella l'immagine .jpg sostituendola con una nera 
    def cancel(self):
        self.QLabelDraw.cancelImage()
        blackimage='C:\\Users\\giovi\\Documents\\GitHub\\ComplexNNproject\\number\\none.jpg'
        self.pixmap = QPixmap(blackimage)
        self.labelImage.setPixmap(self.pixmap)
    
    #inizializzazione dei layout e label che compongono l'applicazione
    def initUI(self):
        
        layout1 = QHBoxLayout()
        layout2 = QVBoxLayout()
        layout3 = QVBoxLayout()

        #label dell'immagine .jpg
        self.labelImage = QLabel(self)

        #label disegnabile 
        self.QLabelDraw = QLabelDraw()
        self.labelImage.setStyleSheet("border: 1px solid black;") 
        blackimage='C:\\Users\\giovi\\Documents\\GitHub\\ComplexNNproject\\number\\none.jpg'
        self.pixmap = QPixmap(blackimage)
        self.QLabelDraw.setFixedSize(400, 400)
        self.QLabelDraw.setStyleSheet("border: 100px solid black;") 
        self.labelImage.setPixmap(self.pixmap)
        self.labelImage.setFixedSize(400, 400)

        layout2.addWidget(self.QLabelDraw)
        layout3.addWidget(self.labelImage)
        
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
        
        layout1.addLayout(layout2)
        layout1.addLayout(layout3)
        
        widget = QWidget(self)
        widget.setLayout(layout1)
        self.setGeometry(0, 0, 825, 450)
        self.setWindowTitle('Number Recognition')
        self.show()

    


def main():
    
    app = QApplication(sys.argv)
    ex = Example()
    


 ######################################################################################
    #Reading
    global nn1
    nn1 = nn([28,18,12])
    numberOfEpochs = 10    # 10

    try:
    #    binary_file_pesi = open('Pesi.bin', mode='rb')   
     #   binary_file_bias = open('Bias.bin', mode='rb') 

      #  my_pesi=pickle.load(binary_file_pesi)
       # nn1.setPesi(my_pesi)
        #my_bias=pickle.load(binary_file_bias)
        #nn1.setBias(my_bias)
        #print("Pesi già presenti!")

        binary_file_pesi = open('Data.bin', mode='rb') 
        t=[]
        for _ in range(2):
           t.append(pickle.load(binary_file_pesi))
        nn1.setPesi(t[0])
        nn1.setBias(t[1])
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
        for x in range(40000):  # numberOfinputs 30000
            targetTrain = int(mydataset.read(1))
            #number = [normalize(float(x)) for x in next(mydataset).split()]
            number = [1 if int(x)>90 else 0 for x in next(mydataset).split()]
            targetsTrain.append(targetTrain)
            inputsTrain.append(number)
        # vettori di targetTrain
        targetVectors = []
        for i in range(len(targetsTrain)):
            targetVectors.append(
                [float(1) if x == targetsTrain[i] else float(0) for x in range(10)])

        mydataset.close()
        print("training terminato")

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


       # tempPesi=pickle.dump(nn1.getPesi(),open('Pesi.bin', 'wb'))
        #tempBias=pickle.dump(nn1.getBias(),open('Bias.bin', 'wb'))

        Data=open('Data.bin', 'wb')
        pickle.dump(nn1.getPesi(),Data)
        pickle.dump(nn1.getBias(),Data)



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

