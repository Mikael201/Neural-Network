
from matplotlib import pyplot as plt
import numpy as np

#PricePerEarnings PE (divided by 5), PricePerBookValue PB, TypeOfCompany(0 = GrowthCompany, 1 = NotAGrowthCompany)
trainingData = [
    [2, 1.0, 1],
    [3, 1.5, 1],
    [6, 2.5, 0],
    [4, 2.5, 0],
    [3.5, 2.0, 1],
    [3.4, 1.0, 1],
    [6.5, 1.5, 0],
    [1, 6.0, 1]
]

#This type of a company will be determined with out Neural Network
PE = float(input("Enter company's PE ratio: "))
PB = float(input("Enter company's PB ratio: "))
unknownData = [PE, PB]

#Neural Network architecture has two inputs and one output. Weight1, Weight2 and Bias b should be trained
w1 = np.random.randn()
w2 = np.random.randn() #w1, w2 and b are first random numbers. With random numbers we first detemine the output
b = np.random.randn()  #After knowing the output with random numbers, we train the w1, w2 and b as long as the output matches our data (0 for GrowthCompany, and 1 for NotAGrowthCompany)


def sigmoid(x): #Sigmoid function is used to scale the output between 0 and 1
    return 1/(1 + np.exp(-x))

def sigmoid_p(x): #Sigmoid function's derivative
    return sigmoid(x) * (1-sigmoid(x))


#loop for training the network.

learning_rate = 0.2

for i in range(50000):
    ri = np.random.randint(len(trainingData))
    point = trainingData[ri]

    z = point[0] * w1 + point[1] * w2 + b #Neural network output Input1 * Weight1 + Input2 * Weight2 + Bias
    pred = sigmoid(z) #Scaling the output between 0 and 1

    target = point[2] #Every output's target is either 0 or 1 (0 = GrowthCompany, 1 = NotAGrowthCompany)
    cost = np.square(pred - target) #CostFunction is our prediction minus our target. Squared.

	#Derivate tells us the speed of the change
    dcost_pred = 2 * (pred - target) #derivate of cost/pred
    dpred_dz = sigmoid_p(z) #derivate of pred/z

    dz_dw1 = point[0] #derivate of z/w1
    dz_dw2 = point[1] #derivate of z/w2
    dz_db = 1 #derivate of z/b

    dcost_dz = dcost_pred * dpred_dz

    dcost_dw1 = dcost_dz * dz_dw1 #partial derivates
    dcost_dw2 = dcost_dz * dz_dw2
    dcost_db = dcost_dz * dz_db

    w1 = w1 - learning_rate * dcost_dw1 #new values w1, w2 and b in our training loop
    w2 = w2 - learning_rate * dcost_dw2
    b = b - learning_rate * dcost_db

print("----------------------------------------------------------")
print("w1, w2 and b are now trained, based on our neural network.") 
print("----------------------------------------------------------")
print("")
print("Now we determine, which companies from our preset data are growth companies and which are not:")
print("")       
for i in range(len(trainingData)):
    point = trainingData[i]
    print(point)
    z = point[0] * w1 + point[1] * w2 + b
    pred = sigmoid(z)
    print("prediction based on the neural network: {}".format(pred))
    if(pred<0.5):
        print("Growth company!")
        print("---------------")
    else:
        print("Not a growth company!")
        print("--------------------")    


print("Unknown data:")
z = unknownData[0] * w1 + unknownData[1] * w2 + b
pred = sigmoid(z)
print("prediction based on the neural network: {}".format(pred))
if(pred<0.5):
    print("The company you entered is statistically A growth company!")
else:
    print("The company you entered is statistically NOT a growth company")    

