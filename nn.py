import numpy as np
from na import na



class nn:
    def __init__(self, myLayers):
        self.myLayers = myLayers
        self.NumberLayers = len(myLayers)
        # definisco primo layer di input
        self.previous = 784
        self.pesi = []
        self.bias=[]
        for x in self.myLayers:
            # inizializzo matrice di pesi randomica
            self.pesi += [2*np.random.random((x, self.previous))-1]
            #inizializzo matrice di byas
            self.bias += [2*np.random.random((x,1))-1]
            # aggiorno layer precedente per definizione matrice pesi successiva
            self.previous = x
        ## inizializzo matrice di bias randomica per layer output
        self.bias += [2*np.random.random((10,1))-1]

        # inizializzo matrice di pesi randomica per layer output
        self.pesi += [2*np.random.random((10, self.previous))-1]
        # print(self.pesi)

    def sigmond(self, x):
        return 1.0/(1.0+np.exp(-x))

    def sigmond_der(self, x):
        return self.sigmond(x)*(1-self.sigmond(x))

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
        a = []
        z = []

        a.append(my_input)
        # adj invertiti
        adj = []
        nablac = []
        delPesi = []
        for x in range(len(self.myLayers) + 1):
            #print(np.dot(self.pesi[0], temp),"\n",self.bias[0],"\n",self.bias[0].T)
            tempz = np.dot(self.pesi[x], np.array(temp))
          #  for t in range(len(tempz)):
          #      tempz[t]=np.add(tempz[t],self.bias[x][t])
            z.append(tempz)
            output = self.sigmond(tempz)
            a.append(output)
            temp = output
        # BP1
        adjL = np.multiply(self.cost_der(
            output, targetVec), self.sigmond_der(z[-1]))
        tempAdj = adjL
        # gli adjustment vanno dal primo all'ultimo
        adj.insert(0, adjL)

        # BP2
        for x in range(1, len(self.myLayers)+1):
            adl = np.multiply(np.dot(self.pesi[-x].T, tempAdj), self.sigmond_der(z[-1-x]))
            #if(x==2):
                #print("HERE MY ADL",adl,  "HERE MY self", z[-1-x])
            tempAdj = adl
            adj.insert(0, adl)





        # BP4
        for x in range(0, len(self.myLayers)+1):
            Del = np.dot(
                np.array([adj[-1-x]]).T, np.array([a[-2-x]]))
            nablac.insert(0, Del)

            #if(x==2):
             #   print(np.array([adj[-1-x]]).T)
        return nablac,adj




    def GradientDescent(self,nabla):
        for x in range(len(self.pesi)):
            self.pesi[x] = self.pesi[x]-nabla[x]


#da aggiungere bias
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
        Sumbias=[]
        for x in range(len(inputTargM)):
            nabla,adj=self.backProp(inpTraining[x],inputTargM[x])
            if(len(Sumdeltanabla)==0):
                Sumdeltanabla=nabla
                Sumbias=adj
            for x in range(len(nabla)):
                Sumdeltanabla[x]=np.add(Sumdeltanabla[x],nabla[x])
                Sumbias[x]=np.add(Sumbias[x],adj[x])



        for x in range(len(self.pesi)):
            self.pesi[x] = self.pesi[x]-(3/10)*Sumdeltanabla[x]
            #self.bias[x] = self.bias[x]-(3/10)*Sumbias[x]
        
















nn1 = nn([12,12])

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
    #number = [1 if int(x)>150 else 0 for x in next(mydataset).split()]
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
#for x in range(30):
#    for j in range(10000):
#        nabla,adj = nn1.backProp(inputsTrain[j], targetVectors[j])
#        nn1.GradientDescent(nabla)



print(nn1.feedforward(inputsTrain[0]))


for j in range(30):
    nn1.TrainNet(inputsTrain, targetVectors)



print(nn1.feedforward(inputsTrain[0]),"\n",targetVectors[0])


