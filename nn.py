import numpy as np


class nn:
    def __init__(self, myLayers):
        self.myLayers = myLayers
        #self.NumberLayers = len(myLayers)
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
        return self.sigmond(x)* (1-self.sigmond(x))

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
        for x in range(len(self.myLayers) + 1):
            tempz = np.dot(self.pesi[x], np.array(temp))
            for t in range(len(tempz)):
                 tempz[t] = tempz[t] + self.bias[x][t]

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
            adl = np.multiply(np.dot(self.pesi[-x].T, tempAdj), self.sigmond_der(np.array(z[-1-x])))
            tempAdj = adl
            adj.insert(0, adl)
        

        # BP4
        for x in range(0, len(self.myLayers)+1):
            Del = np.dot(
                np.array([adj[-1-x]]).T, np.array([a[-2-x]]))
            nablac.insert(0, Del)

        return nablac,adj




    def GradientDescent(self,nabla):
        for x in range(len(self.pesi)):
            self.pesi[x] = self.pesi[x]-nabla[x]



    def feedforward(self, my_input):
        temp = my_input
        for x in range(len(self.myLayers) + 1):
            tempz = np.dot(self.pesi[x], np.array(temp))
            for t in range(len(tempz)):
                tempz[t]=np.add(tempz[t],self.bias[x][t])
            output = self.sigmond(tempz)
            temp = output
        return output



    def TrainNet(self,inputTr,inputTrag):   
        #minibatch da 10
        self.mb_size = 10
        list_of_inp = zip(*(iter(inputTr),) * self.mb_size)
        list_of_targ = zip(*(iter(inputTrag),) * self.mb_size)
        list_Global = zip(list_of_inp,list_of_targ)

       

        for x in list_Global : 
            self.minibatchUpd(x[0],x[1])





    def minibatchUpd(self,inpTraining,inputTargM):
        Sumdeltanabla=[]
        Sumbias=[]
        for j in range(len(inputTargM)):   # len(inputTargM) == 10
            nabla,adj=self.backProp(inpTraining[j],inputTargM[j])
            if(len(Sumdeltanabla)==0):
                Sumdeltanabla=nabla
                Sumbias=adj
            #print(adj[0]) #??????
            for x in range(len(nabla)):
                Sumdeltanabla[x]=np.add(Sumdeltanabla[x],nabla[x])   # perchè sommi i delta ai deltanabla?
                Sumbias[x]=np.add(Sumbias[x],adj[x])   


        self.eta = 5
       
        for x in range(len(self.pesi)):   
            for z in range(len(Sumdeltanabla[x])):
                self.pesi[x][z] = self.pesi[x][z]-(self.eta/self.mb_size)*Sumdeltanabla[x][z]
            for z in range(len(Sumbias[x])):
                self.bias[x][z]=self.bias[x][z]-(self.eta/self.mb_size)*Sumbias[x][z]


    
        



nn1 = nn([12,12])

####################################################
# input dal file

targets = []
inputsTrain = []
#mydataset = open("data/MnistTrain.txt", "r")
mydataset = open(r"C:\\Users\\bigfo\\OneDrive\\Desktop\\dati\\mnistTrain_copy.txt", "r")
for x in range(30000):
    target = int(mydataset.read(1))
    #number = [int(x) for x in next(mydataset).split()]
    number = [1 if int(x)>90 else 0 for x in next(mydataset).split()]
    targets.append(target)
    inputsTrain.append(number)

# vettori di target
targetVectors = []
for i in range(len(targets)):
    targetVectors.append(
        [float(1) if x == targets[i] else float(0) for x in range(10)])


mydataset.close()

####################################################




for x in range(10):
    nn1.TrainNet(inputsTrain, targetVectors)


for x in range(len(inputsTrain)):
    output=nn1.feedforward(inputsTrain[x])
    print("output desiderato: ",targets[x])
    max=0
    count=0
    for z in range(len(output)):
        if output[z]>max:
            max=output[z]
            count=z
    print("output ottenuto: ",count)






