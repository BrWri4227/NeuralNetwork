import numpy as np
import csv
import random
import matplotlib.pyplot as plt

print("Importing Data...") 
with open('mnist_train.csv', 'r') as file1: #READ IN FILE
    csvReader = csv.reader(file1)
    train_data = list(csvReader)[1:] 

with open('mnist_test.csv', 'r') as file2: # READ IN OTHER FILE
    csvReader = csv.reader(file2)
    test_data = list(csvReader)[1:] 
    
train_labels = np.array([int(row[0]) for row in train_data]) #Pull labels off csv
train_data = np.array([row[1:] for row in train_data], dtype=np.float64) #pull data out of csv
train_data = train_data / 255 #normalize because if i didn't it makes sigmoid go brazy
validation_labels = np.array([int(row[0]) for row in test_data]) #SAME THING BUT OTHER FILE
validation_data = np.array([row[1:] for row in test_data], dtype=np.float64)
validation_data = validation_data / 255

allData = np.concatenate((train_data, validation_data)) #Put em together
allLabels = np.concatenate((train_labels, validation_labels)) #combine em

random.seed(42)
shuffleIndex = np.arange(len(allData)) #generate index so i can align labels and data
np.random.shuffle(shuffleIndex)  #shuffle index
allData = allData[shuffleIndex]  #shuffle data
allLabels = allLabels[shuffleIndex] #shuffle labels

trainSet = allData[:42000] #SPLIT DATA INTO TRAINING AND VALIDATION
trainLabels = allLabels[:42000] #60
validationSet = allData[42000:56000] #20
validationLabels = allLabels[42000:56000] #20
testSet = allData[56000:] #20 
testLabels = allLabels[56000:] #20
weights = [] #init weights
bias = [] # init bias
testAccs = [] #init test accs
validationAccs = [] #init validation accs



learnRate = 0.01 #learning rate!
inputSize = 784 #init network params 
outputSize = 10
hiddenLayerCount = 3 #hidden layer count, change for more or less
hiddenLayerSize = 100
epochs = 3#epoch count


print("Finished importing Data...")


def initializeNetwork(inputSize, outputSize, hiddenLayerCount, hiddenLayerSize):    
    if hiddenLayerCount == 0: #if no hidden layer
        weights.append(np.random.uniform(-0.5, 0.5, (inputSize, outputSize))) #make output layer
        bias.append(np.zeros(outputSize)) #init bias
    else: #else it has at least one hidden layer
        for i in range(hiddenLayerCount): #for each hidden layer
            if i == 0:
                weights.append(np.random.uniform(-0.5, 0.5, (inputSize, hiddenLayerSize))) #make first hidden layer (has to get from input layer to hidden layer)
                bias.append(np.zeros(hiddenLayerSize))
            elif i == hiddenLayerCount - 1:
                weights.append(np.random.uniform(-0.5, 0.5, (hiddenLayerSize, outputSize))) #make last hidden layer (has to get from last hidden layer to output layer)
                bias.append(np.zeros(outputSize))
            else:
                weights.append(np.random.uniform(-0.5, 0.5, (hiddenLayerSize, hiddenLayerSize))) #else hidden layer to hidden layer
                bias.append(np.zeros(hiddenLayerSize))
    return weights, bias

def calcloss(predictions, actuals): #cross entropy loss calculation 
    return -np.sum(predictions * np.log(actuals))

def sigmoid(x): #activation func
    return 1/(1+np.exp(-x))

def sigmoid_Derivative(x): #derivative of activation func
    return sigmoid(x) * (1 - sigmoid(x))

def forwardProp(x): #forward prop
    output = x
    outputs = [output]
    for i in range(len(weights)): #for each layer
        output = np.dot(output, weights[i]) + bias[i] #multiply by weights and add bias
        output = sigmoid(output) #apply activation func
        outputs.append(output) #add to outputs
    return outputs

def backProp(y, predictions): #back prop
    targets = np.zeros_like(predictions[-1]) #init target values
    targets[y] = 1
    error = predictions[-1] - targets #error is the difference between the prediction and the target 
    for i in reversed(range(len(weights))): #for each layer
        if i == len(weights) - 1: #if its the last layer
            delta = (predictions[-1] - targets) * sigmoid_Derivative(predictions[i+1])#its the output layer
        else:
            delta = error * sigmoid_Derivative(predictions[i+1]) #other wise its a hidden layer
        deltaWeights = np.outer(predictions[i].T, delta) #calculate delta weights
        deltaBias = np.sum(delta, axis=0) #calculate delta bias
        weights[i] -= learnRate * deltaWeights #update weights
        bias[i] -= learnRate * deltaBias #update bias
        error = np.dot(delta, weights[i].T) #update error
        
def train(trainSet,trainLabels, epochs): #training
    
    for epoch in range(epochs): #epoch counter
        for i in range(len(trainSet)): #for each training example
            predictions = forwardProp(trainSet[i]) #forward prop
            backProp(trainLabels[i], predictions) #bacprop the result
        ValidationAcc = evaluate(validationSet, validationLabels) #each epoch evaluate the validation set
        testAcc = evaluate(testSet, testLabels) #each epoch evaluate the test set
        print("Epoch: ", epoch + 1, "Validation Accuracy: {:.2f}%".format(ValidationAcc*100), "Test Accuracy: {:.2f}%".format(testAcc * 100) )
        validationAccs.append(ValidationAcc) #add validation acc to list
        testAccs.append(testAcc)
    
def test(testSet, testLabels): #test
    correct = 0 #init correct
    for i in range(len(testSet)): #for each test example
        predictions = forwardProp(testSet[i]) #forward prop
        prediction = predictions[-1] #get prediction (last layer)
        if np.argmax(prediction) == testLabels[i]: #if prediction is correct
            correct += 1 #add to correct
    return correct/len(testSet) #return accuracy
def evaluate(validationSet, validationLabels): #SAME THING BUT DIFFERENT SET, IDK WHY I MADE TWO, SEPERATES THEM IN MY MIND
    correct = 0
    for i in range(len(validationSet)):
        predictions = forwardProp(validationSet[i])
        prediction = predictions[-1]
        # print("Prediction: ", np.argmax(prediction), "Actual: ", validationLabels[i])
        if np.argmax(prediction) == validationLabels[i]:
            correct += 1
    return correct/len(validationSet)




print("Initializing network...")
weights, bias = initializeNetwork(inputSize, outputSize, hiddenLayerCount, hiddenLayerSize) #init network
print("Training network...")
train(trainSet, trainLabels, epochs) #train network
test(testSet, testLabels)#test network
epochRanges = list(range(1, epochs+1)) #create epoch range list
plt.plot(epochRanges,testAccs,'ro-' ,label='Test Accuracy', ) #plot test acc
plt.plot(epochRanges,testAccs,'ro-' ,label='Test Accuracy', ) #plot test acc
plt.plot(epochRanges,validationAccs,'bo-', label='Validation Accuracy') #plot validation acc
plt.ylabel('Accuracy') #label plot
plt.xlabel('Epoch')
plt.legend()
plt.title('Accuracy vs Epoch') #title plot
plt.show() #show plot
plt.savefig('Accuracy vs Epoch.png') #save plot
