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


    def MSE(self, targets) :
        m = 1
        totalSum = 0
        # dovremo ricordarci di modificare if x == targets[i] in modo da avere 
        # ogni volta i valori di una batch successiva, dopo che avremo suddiviso gli input in batches
        for i in range(m) :
            targetVector = [float(1) if x == targets[i] else float(0) for x in range(10)]
            #print(targets[i], targetVector)
            totalSum += 0.5*np.linalg.norm(self.outputs[i] - targetVector)**2
       # print(targetVector, "\n", self.outputs[0])

        return totalSum / m




    def backProp(self, my_input,targetVec):
        temp = my_input
        self.a=[]
        self.z=[]


        #adj invertiti
        self.adj=[]
        #self.delPesi=[np.zeros(w.shape) for w in self.pesi]
        self.delPesi=[]
        for x in range(len(self.myLayers) + 1):
            tempz=np.dot(self.pesi[x], temp)
            self.z.append(tempz)
            self.output = self.sigmond(tempz)
            self.a.append(self.output)
            temp = self.output
        #BP1
        self.adjL=np.multiply(self.cost_der(self.output,targetVec),self.sigmond_der(self.z[len(self.z)-1]))

        tempAdj=self.adjL
        #BP2
        for x in range(1,len(self.myLayers)+1):
                adl=np.multiply(np.dot(self.pesi[len(self.pesi)-x].T,tempAdj),self.sigmond_der(self.z[len(self.z)-1-x]))
                tempAdj=adl
                self.adj.append(adl)
        #BP4 layer finale
        LastDel=np.dot(self.adjL,self.a[len(self.a)-2].T)
        print(LastDel)
        #d=np.dot(self.a[],self.adjL)



        return self.output


nn1 = nn([12, 12])

####################################################
# input dal file

targets = []
inputsTrain = []
#mydataset = open("data/MnistTrain.txt", "r")
mydataset = open(r"C:\\Users\\bigfo\\OneDrive\\Desktop\\dati\\mnistTrain_copy.txt", "r")
for x in range(10000):
    target = int(mydataset.read(1))
    number = [int(x) for x in next(mydataset).split()]
    targets.append(target)
    inputsTrain.append(number)

#vettori di target
targetVectors=[]
for i in range(len(targets)):
    targetVectors.append([float(1) if x == targets[i] else float(0) for x in range(10)])



mydataset.close()

####################################################
# feedforward
#for i in range(len(inputsTrain)):
nn1.backProp(inputsTrain[0],targetVectors[0])



minibatchesSize = 10
minibatch = inputsTrain[:minibatchesSize]

#print("COST: ", nn1.MSE(targets))
