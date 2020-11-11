import numpy as np
from na import na


class nn:
    def __init__(self, myLayers):
        self.myLayers = myLayers
        self.NumberLayers = len(myLayers)
        # definisco primo layer di input
        self.previous = 784
        self.pesi = []
        self.outputs = []
        for x in self.myLayers:
            # inizializzo matrice di pesi randomica
            self.pesi += [2*np.random.random((x, self.previous))-1]
            # aggiorno layer precedente per definizione matrice pesi successiva
            self.previous = x
        # inizializzo matrice di pesi randomica per layer output
        self.pesi += [2*np.random.random((10, self.previous))-1]
        # print(self.pesi)

    def sigmond(self, x):
        return 1/(1+np.exp(-x))

    def sigmond_der(self, x):
        return x*(1-x)

    def getPesi(self):
        return self.pesi

    def MSR(self) :
        pass


    def TrainNN(self, nTimes):
        pass

    def getOutput(self, my_input):
        temp = my_input
        for x in range(len(self.myLayers) + 1) :
            output = self.sigmond(np.dot(self.pesi[x], temp))
            temp = output
        return output




nn_prova = nn([12, 12])
# print(x.getPesi())

target = []
inputT = []
#mydataset = open("data/MnistTrain.txt", "r")
mydataset = open(r"C:\\Users\\bigfo\\OneDrive\\Desktop\\dati\\mnistTrain_copy.txt", "r")
for x in range(10000):
    target.append(int(mydataset.read(1)))
    inputT.append([int(x) for x in next(mydataset).split()])

mydataset.close()

# print(pesi_sinaptici)


#print(target[0]," ",inputT[0], len(inputT[0]))

# a=np.array([inputT[0]]).T

a = np.array(inputT[0])



print(nn_prova.getOutput(inputT[0]))

# a' = W a
