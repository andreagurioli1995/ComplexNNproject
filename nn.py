import numpy as np
class nn:
    def __init__(self,x):
        self.x=x
    def getX(self):
        return self.x



x= nn(1)

output=[]
inputT=[]
mydataset = open(r"C:\\Users\\bigfo\\OneDrive\\Desktop\\dati\\mnistTrain_copy.txt","r")
for x in range(10000):
    output.append(int(mydataset.read(1)))
    temp=next(mydataset).split()
    for x in range(len(temp)):
       temp[x]=int(temp[x])
    inputT.append(temp)



print(output[0]," ",inputT[0], len(inputT[0]))

a=np.array(inputT[0]).T
print(a)

#trasposta
#training_outputs= np.array([[0,1,1,0]]).T

#moltiplicazione riga per colonna
#pesi_sinaptici+= np.dot(input_layer.T,adjustment)

#random
#np.random.seed(1)
#pesi_sinaptici = 2*np.random.random((3,1))-1


#def sigmond(x):
#    return 1/(1+np.exp(-x))
#def sigmond_der(x):
#    return x*(1-x)

#commento prova
