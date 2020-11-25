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

    def cost_der(self, output_activations, y):
        return (output_activations-y)

    def getPesi(self):
        return self.pesi

    def MSE(self, targets):
        m = 1
        totalSum = 0
        # dovremo ricordarci di modificare if x == targets[i] in modo da avere
        # ogni volta i valori di una batch successiva, dopo che avremo suddiviso gli input in batches
        for i in range(m):
            targetVector = [float(1) if x == targets[i]
                            else float(0) for x in range(10)]
            #print(targets[i], targetVector)
            totalSum += 0.5*np.linalg.norm(self.outputs[i] - targetVector)**2
       # print(targetVector, "\n", self.outputs[0])

        return totalSum / m

    def backProp(self, my_input, targetVec):
        temp = my_input
        self.a = []
        self.z = []

        self.a.append(my_input)
        # adj invertiti
        self.adj = []
        self.nablac = []
        #self.delPesi=[np.zeros(w.shape) for w in self.pesi]
        self.delPesi = []
        for x in range(len(self.myLayers) + 1):
            tempz = np.dot(self.pesi[x], temp)
            self.z.append(tempz)
            self.output = self.sigmond(tempz)
            self.a.append(self.output)
            temp = self.output
        # BP1
        self.adjL = np.multiply(self.cost_der(
            self.output, targetVec), self.sigmond_der(self.z[-1]))
        tempAdj = self.adjL

        # gli adjustment vanno dall'ultimo al primo
        self.adj.insert(0, self.adjL)

        # BP2
        for x in range(1, len(self.myLayers)+1):
            adl = np.multiply(np.dot(self.pesi[-x].T, tempAdj), self.sigmond_der(self.z[-1-x]))
            tempAdj = adl
            self.adj.insert(0, adl)

        # BP4 layer finale
        #LastDel=np.array([self.adjL]).T @ np.array([self.a[len(self.a)-2]])
        #self.pesi[-1]=self.pesi[-1] - LastDel
        # print(self.pesi[-1])

        # BP4
        for x in range(0, len(self.myLayers)+1):
            Del = np.dot(
                np.array([self.adj[-1-x]]).T, np.array([self.a[-2-x]]))
            self.nablac.insert(0, Del)
        return self.nablac




    def GradientDescent(self):
        for x in range(len(self.pesi)):
            self.pesi[x] = self.pesi[x]-self.nablac[x]

    def feedforward(self, my_input):
        temp = my_input
        for x in range(len(self.myLayers) + 1):
            output = self.sigmond(np.dot(self.pesi[x], temp))
            temp = output
        return output





    def TrainNet(self,inputTr,inputTrag):
        #minibatch da 10
        list_of_inp = zip(*(iter(inputTr),) * 10)
        list_of_targ = zip(*(iter(inputTrag),) * 10)
        list_Global= zip(list_of_inp,list_of_targ)
        for x in list_Global:
          self.minibatchUpd(x[0],x[1])





    def minibatchUpd(self,inpTraining,inputTargM):
        Sumdeltanabla=[]
        for x in range(len(inputTargM)):
            Sumdeltanabla+=self.backProp(inpTraining[x],inputTargM[x])


        for x in range(len(self.pesi)):
            self.pesi[x] = self.pesi[x]-(0.000000000001/10)*Sumdeltanabla[x]














nn1 = nn([12, 12])

####################################################
# input dal file

targets = []
inputsTrain = []
#mydataset = open("data/MnistTrain.txt", "r")
mydataset = open(
    r"C:\\Users\\bigfo\\OneDrive\\Desktop\\dati\\mnistTrain_copy.txt", "r")
for x in range(10000):
    target = int(mydataset.read(1))
    number = [int(x) for x in next(mydataset).split()]
    targets.append(target)
    inputsTrain.append(number)

# vettori di target
targetVectors = []
for i in range(len(targets)):
    targetVectors.append(
        [float(1) if x == targets[i] else float(0) for x in range(10)])


mydataset.close()

####################################################
# feedforward
#for j in range(5):
#    nn1.backProp(inputsTrain[j], targetVectors[j])
#    nn1.GradientDescent()



print(nn1.feedforward(inputsTrain[0]))

for j in range(2):
    nn1.TrainNet(inputsTrain, targetVectors)


print(nn1.feedforward(inputsTrain[0]))


