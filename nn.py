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

<<<<<<< HEAD
    def setOutputs(self, inputs) :
        for x in range(100) :
            self.outputs.append(self.feedForward(inputs[x]))

    def MSR(self, targets) :
        m = 10
        totalSum = 0
        # dovremo ricordarci di modificare if x == targets[i] in modo da avere 
        # ogni volta i valori di una batch successiva, dopo che avremo suddiviso gli input in batches
        for i in range(m) :
            targetVector = [float(1) if x == targets[i] else float(0) for x in range(10)]
            #print(targets[i], targetVector)
            totalSum = np.linalg.norm(self.outputs[i] - targets[i])**2
        return totalSum / m
=======
    def MSR(self) :
        pass

>>>>>>> ab82e6f17c2054e1e033dce1d56b4b48b8735b8e

    def TrainNN(self, nTimes):
        pass

    def feedForward(self, my_input):
        temp = my_input
        for x in range(len(self.myLayers) + 1) :
            output = self.sigmond(np.dot(self.pesi[x], temp))
            temp = output
        return output


nn1 = nn([12, 12])

####################################################
# input dal file

<<<<<<< HEAD
targets = []
inputsTrain = []
mydataset = open("data/MnistTrain.txt", "r")
=======
nn_prova = nn([12, 12])
# print(x.getPesi())

target = []
inputT = []
#mydataset = open("data/MnistTrain.txt", "r")
mydataset = open(r"C:\\Users\\bigfo\\OneDrive\\Desktop\\dati\\mnistTrain_copy.txt", "r")
>>>>>>> ab82e6f17c2054e1e033dce1d56b4b48b8735b8e
for x in range(10000):
    target = int(mydataset.read(1))
    number = [int(x) for x in next(mydataset).split()]
    targets.append(target)
    inputsTrain.append(number)

mydataset.close()

####################################################
# feedforward

nn1.feedForward(inputsTrain[0])
nn1.setOutputs(inputsTrain)

minibatchesSize = 10
minibatch = inputsTrain[:minibatchesSize]

print("COST: ", nn1.MSR(targets))
print("\n\n\n\n\n")