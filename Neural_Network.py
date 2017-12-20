#note: in testing, i placed this file in the same location as the mnist training and test files
#alternatively, you can set the file locations in the train and test portions of __main__
#run using command - python CS156_FinalProject.py

import numpy as np


class ANN(object):


    def __init__(self, inputnumber, hiddennumber, outputnumber, lrate):

        #set variables from constructor params
        self.learningrate = lrate
        self.inputnodes = inputnumber
        self.hiddennodes = hiddennumber
        self.outputnodes = outputnumber

        #set standard deviation for normal distribution
        #inverse square root of number of nodes in output for normal distribution
        self.layer1scale = pow(self.hiddennodes,-0.5)
        self.layer2scale = pow(self.outputnodes,-0.5)

        #set random weights array
        #dimensions of array are determined by number of nodes layers
        self.w_layer1 = np.random.normal(0.0, self.layer1scale, (self.hiddennodes, self.inputnodes))
        self.w_layer2 = np.random.normal(0.0, self.layer2scale, (self.outputnodes, self.hiddennodes))

    def train(self, inputvals, targetvals): #this function performs forward and back propagation

        #create 2-dimensional input array and take transpose
        inputs = np.array(inputvals, ndmin=2).T

        #hidden inputs - dot product of the weights and the inputs
        hinputs = np.dot(self.w_layer1, inputs)

        #hidden outputs (inputs to the next set of calculations) - sigmoid function is applied
        houtputs = self.sigmoid(hinputs)

        #inputs to the next set of nodes are calculated (weights by inputs from prev layer)
        hinputs2 = np.dot(self.w_layer2, houtputs)

        #outputs take the inputs from the prev calculation and apply the sigmoid function
        outputs = self.sigmoid(hinputs2)

        #create a 2-dimensional target array and take transpose
        targets = np.array(targetvals, ndmin=2).T

        #ERROR calcualtions:
        #compute error for output layer
        oerrors = targets - outputs

        #compute error for hidden layer
        herrors = np.dot(self.w_layer2.T, oerrors)

        #update weight arrays based on errors
        intrans = inputs.T
        self.w_layer1 += self.learningrate*np.dot(herrors*houtputs*(1-houtputs), intrans)
        htrans = houtputs.T
        self.w_layer2 += self.learningrate*np.dot(oerrors*outputs*(1-outputs), htrans)

    def test(self, inputvals):

        #create input array using the input values
        inputs = np.array(inputvals, ndmin=2).T

        #take dot product of weights and input values to produce inputs to hidden layer
        hinputs = np.dot(self.w_layer1, inputs)

        #apply activation function to the hinputs array
        houtputs = self.sigmoid(hinputs)

        #create input array for final activation using outputs from hidden layer
        hinputs2 = np.dot(self.w_layer2, houtputs)

        #apply activation function to the hinputs2 array - generate final outputs
        outputs = self.sigmoid(hinputs2)

        return outputs

    #define the sigmoid (activation) function
    def sigmoid(self, x):
        return 1/(1+np.exp(-x))



if __name__ == '__main__':

    #Initialize
    #------------------------------------------
    #set neural network parameters
    inputnumber = 784 #number of pixels
    hiddennumber = 200
    outputnumber = 10 #number of output options
    lrate = 0.125 #learning rate adjusted based on testing - can be changed to see how accuracy changes

    #instantiate a nueral network using the set paramters
    testnetwork = ANN(inputnumber,hiddennumber,outputnumber,lrate)

    #Train
    #------------------------------------------
    #open train file, save data, close train file
    tfile = open('mnist_train.csv','r')
    temp = tfile.readlines()
    tfile.close()

    for row in temp:
        vals = row.split(',') #create array of comma separated values
        inputs = 0.01+(np.asfarray(vals[1:])/255.0*0.99) #create float array of the scaled input values (pixels), skip first value which is target
        targets = 0.01+(np.zeros(outputnumber)) #create 10 value target array, all values 0.01
        targets[int(vals[0])]=0.99 #set the weight of the target index to .99
        testnetwork.train(inputs, targets) #train network using the input and target arrays

    #Test
    #------------------------------------------
    #open test file, save data, close test file
    tfile2 = open('mnist_test.csv','r')
    temp = tfile2.readlines()
    tfile2.close()

    #initialize count and total, used to determine the percent accuracy (count/total * 100 = percent accuracy)
    count = 0.0
    total = 0.0

    for row2 in temp:
        testvals = row2.split(',') #split string of comma separated values into an array, similar to explode in php
        correct = int(testvals[0])  #set correct value
        inputs = 0.01+(np.asfarray(testvals[1:])/255.0*.99) #create float array of the scaled input values (pixels), skip first value which is target
        outputs = testnetwork.test(inputs)

        #increment total for each test case
        total+=1

        #assign the label to be the highest probability output from the neural network
        label = np.argmax(outputs)

        #if the neural network guessed correctly, increment the count
        if label == correct:
            count+=1

    #Results (percent accuracy)
    #------------------------------------------
    #print the percent accuracy based on the number of correct predictions
    srate = float(count)/float(total)
    print 'Percent Accuracy:',srate*100,'%'
