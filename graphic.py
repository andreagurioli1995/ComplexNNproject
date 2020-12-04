from PyQt5.QtWidgets import QMainWindow, QApplication ,QLabel, QMenu, QFrame, QMenuBar,QHBoxLayout,QVBoxLayout,QAction, QFileDialog, QWidget, QPushButton
from PyQt5.QtGui import QIcon, QImage, QPainter, QPen, QBrush,QPixmap
from PyQt5.QtCore import Qt, QPoint
import cv2
import config 
import numpy
import sys

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
        self.brushSize = 35
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
                self.value=self.img.pixel(i,j)%256
                self.mnist.append(self.value)
        #Example().set_mnist(self.mnist)
        #print(self.mnist)
        self.update()
        config.x=self.mnist
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
        print(config.x)
      
       
    
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
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
    