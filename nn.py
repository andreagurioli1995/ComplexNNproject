import numpy as np
class nn:
    def __init__(self,x):
        self.x=x
        print(x)
    def getX(self):
        return self.x



x= nn(1)
mydataset = open(r"C:\\Users\\bigfo\\OneDrive\\Desktop\\dati\\mnistTrain_copy.txt","r")
a= mydataset.read
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

