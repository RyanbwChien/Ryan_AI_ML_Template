import numpy as np
from random import seed
from random import random
from math import exp

# Initialize a network
def initialize_network(n_inputs, n_hidden):
    network = list()
    hidden_layer1 = np.array([random() for i in range((n_inputs + 1)*n_hidden)]).reshape(n_hidden,(n_inputs + 1))
    network.append(hidden_layer1)
    return network
# Initialize a network
def initialize_network_output(n_hidden, n_outputs):
    network = list()
    output_layer = np.array([random() for i in range((n_hidden + 1)*n_outputs)]).reshape(n_outputs,(n_hidden + 1))
    network.append(output_layer)
    return network

# sigmoid activation
def sigmoid(activation):
	return np.array([1.0 / (1.0 + exp(-activation[i])) for i in range(len(activation))])
# linear activation
def linear(activation):
	return np.array([activation[i] for i in range(len(activation))])
# softmax activation
def softmax(activation):
    exps = np.exp(activation)
    return exps / np.sum(exps)
# ReLU activation
def ReLU(activation):
	return np.array([max(0,activation[i]) for i in range(len(activation))])

# Forward propagate input to a network output
# 一次對"一個個體的" X P個特徵

def forward_propagate(model, x):
    inputs = np.append(1,x)  # Add 1 in input vector 
    inneroutput=[]
    count=1
    for layer in model:
        summation = layer.dot(np.array(inputs))
        if(count!=len(model)):
            a=sigmoid(summation)
            count+=1
        else:
            a=softmax(summation)
        inneroutput.append(a)
        inputs = np.append(1,a)
    return [inputs[1:],inneroutput]


# Loss function 
def squarelossfunction(y,yhat):
    sse=0
    for i in range(len(yhat)):
        sse+=(yhat[i] - y[i])**2
    mse=1/len(y)*sse
    return  (mse)
def crossentropyfunction(y,yhat):
    result=[]
    for i in range(len(yhat)):
        result.append(-np.array(y[i,:]).T.dot(np.log(yhat[i,:])))
    return (result)   

# Loss function derivative
def lossfunction_derivative(y,yhat):
	return  np.array([yhat[i] - y[i] for i in range(len(yhat))] )
def entropy_softmax_derivative(y,yhat):
	return  np.array([yhat[i] - y[i] for i in range(len(yhat))] )

# Activity function derivative
def softmax_derivative(output):
	return np.array([output[i] * (1.0 - output[i]) for i in range(len(output))])
def linear_derivative(output):
	return np.array([1 for i in range(len(output))])
def corresmulti(x,y):
    return np.array([x[i]*y[i] for i in range(len(x))]).reshape(-1,1)


# Backpropagate error and store in neurons for one indiviuals
def backward_propagate_error(model, y, yhat,x):
    #dL_div_dyhat=lossfunction_derivative(y,yhat[0])
    #dyhat_div_dzo=linear_derivative(yhat[0])
    end=entropy_softmax_derivative(y,yhat[0])
    end=np.array(end)
    dL_div_weight = list()   
    temp=[]
    for i in reversed(range(len(model))):
		#layer = network[i]
        temp=[]
        if i==len(model)-1:  # Btw hidden layer and output layer weight
            for j in end.reshape(-1,1):
                temp.append(np.append(1,yhat[1][i-1])*j) # Btw hidden layer and output layer weight (K+1)*O
            dL_div_weight.append(np.array(temp)) 
 
            dz_div_da=model[i].T[1:,:].dot(end.reshape(-1,1)) # dz_div_da
            da_div_dz=softmax_derivative(yhat[1][i-1]) # da_div_dz
            end=corresmulti(dz_div_da , da_div_dz)

        if (i!=0 and i!=len(model)-1): # Btw k-th hidden layer and (k-i)-th output layer weight
            for j in end.reshape(-1,1):
                temp.append(np.append(1,yhat[1][i-1])*j) # Btw k-th hidden layer and (k-i)-th output layer weight
            dL_div_weight.append(np.array(temp))
            dz_div_da=model[i].T[1:,:].dot(end.reshape(-1,1)) # dz_div_da
            da_div_dz=softmax_derivative(yhat[1][i-1]) # da_div_dz
            end=corresmulti(dz_div_da , da_div_dz)
    
        if i==0: # Btw 1st hidden layer and input layer weight
            for jj in end.reshape(-1,1):
                temp.append(np.append(1,x)*jj)
            dL_div_weight.append(np.array(temp))
    return(dL_div_weight)

# gradient descent method 
# batch_size=1 means SGD
# batch_size>1 <n means Mini batch GD   
# batch_size=n means GD   
batch_size=10
def MBGD(x,model,y,batch_size):
    total=[]
    idx=np.random.choice(len(x), size=batch_size, replace=False)
    #idx=np.random.randint(0,len(x), size=batch_size)
    count=0
    #idx=np.random.randint(0,len(x), size=batch_size)
    for j in range(len(model)):  
        for i in idx:
            yhat=forward_propagate(model, x[i,:])
            dL_div_weight=backward_propagate_error(model, y[i], yhat,x[i,:])

            if count==0:
                a=np.array(dL_div_weight[j])
            else:
                a=np.array(dL_div_weight[j])+a
            count+=1
        count=0
        total.append(1/len(x)*a)
    return (total)

def train_network_MBGD(model, train, l_rate, n_epoch,y,batch_size):
    for epoch in range(n_epoch):
        for j in range(len(model)):  
            model[j]+= -(l_rate * MBGD(x,model,y,batch_size)[len(model)-1-j])
    #print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
    return(model)

# Make a prediction with a network
def predict(anetwork, x):
	outputs = [forward_propagate(anetwork, x[i,:])[0] for i in range(len(x))]
	return outputs 

def correctrate(y , yyhat):
    sumt1=[]
    sumt2=[]
    sumt3=[]
    for i in range(np.size(y,0)):
        sumt1.append(np.where(y[i,:]==max(y[i,:])))
        sumt2.append(np.where(yyhat[i,:]==max(yyhat[i,:])))
        sumt3
    result=sum([sumt1[i]==sumt2[i] for i in range(len(sumt1))])
    return(result)

# Generate normal random dataset
seed(1)
mu=[5,6,7,8,9]
sigma =[5,4,3,2,1]

def genernormal(mu,sigma,n,k):
    output =np.array([np.random.choice( np.append(1,np.repeat(0,k-1)) , size=k, replace=False, p=None) for i in range(n)])
    for i in range(len(mu)):
        if i==0:
            data=np.random.normal(mu[i], sigma[i],size=(n,1))      
        else:
            data=np.concatenate(  (data,np.random.normal(mu[i], sigma[i],size=(n,1))),axis=1 )
    return([data,output])

def normallization(x):
    for i in range(np.size(x,1)):
        for j in range(np.size(x,0)):
            x[j,i]=(x[j,i]-min(x[:,i]))/(max(x[:,i])-min(x[:,i]))
    return(x)


# =============================================================================
# Build model
# =============================================================================

# User defined #input /hidden / output layer count
# number of output need same as the number of y classification

model=[]
model.append(initialize_network(5, 10)[0])
#model.append(initialize_network(10, 10)[0])
#model.append(initialize_network(6, 5)[0])
#model.append(initialize_network(10, 10)[0])
#model.append(initialize_network(10, 10)[0])
model.append(initialize_network_output(10, 5)[0])


x,y= genernormal(mu,sigma,n=100,k=5)
x=normallization(x)
yyhat=np.array(predict(model, x))
sum_error=np.sum(crossentropyfunction(y,yyhat))
correctrate(y , yyhat)/len(y)

anetwork=train_network_MBGD(model, x,0.15,1000 , y ,100) 
yyhat=np.array(predict(anetwork, x))
meantotalloss=np.sum(crossentropyfunction(y,yyhat))

np.apply_along_axis(sum, 0,y)
correctrate(y , yyhat)/len(y) # Value more close to 1 means classification more correctly

