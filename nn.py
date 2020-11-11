import numpy as np
from na import na
class nn:
    def __init__(self,myLayers):
        self.myLayers=myLayers
        self.NumberLayers= len(myLayers)
        #definisco primo layer di input
        self.previous = 784
        self.pesi=[]
        for x in self.myLayers:
            #inizializzo matrice di pesi randomica
            self.pesi+=[2*np.random.random((x,self.previous))-1]
            #aggiorno layer precedente per definizione matrice pesi successiva
            self.previous=x
        #inizializzo matrice di pesi randomica per layer output
        self.pesi+=[2*np.random.random((10,self.previous))-1]
        #print(self.pesi)


    def sigmond(self,x):
        return 1/(1+np.exp(-x))

    def sigmond_der(self,x):
        return x*(1-x)

    def getPesi(self):
        return self.pesi

    def TrainNN(self,nTimes):
        pass

    def getOutput(self,my_input):
        pass


x= nn([12,12])
print(x.getPesi())

output=[]
inputT=[]
mydataset = open("data/MnistTrain.txt", "r")
for x in range(10000):
    output.append(int(mydataset.read(1)))
    inputT.append([int(x) for x in next(mydataset).split()])

#print(pesi_sinaptici)


print(output[0]," ",inputT[0], len(inputT[0]))

a=np.array([inputT[0]]).T
#print(a)

#trasposta
#training_outputs= np.array([[0,1,1,0]]).T

#moltiplicazione riga per colonna
#pesi_sinaptici+= np.dot(input_layer.T,adjustment)

#random
#np.random.seed(1)
#pesi_sinaptici = 2*np.random.random((3,1))-1




#alternativa alla funzione di sigmund:
# rectified linear function
#def rectified(x):
#	return max(0.0, x)
#"""https://machinelearningmastery.com/rectified-linear-activation-function-for-deep-learning-neural-networks/"""
