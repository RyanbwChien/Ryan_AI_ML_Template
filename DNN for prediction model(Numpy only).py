
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

# Transfer neuron sigmoid activation
#activation=summation
#i=1
def sigmoid(activation):
	return np.array([ (1.0 / (1.0 + exp(-activation[i]))) for i in range(len(activation))])
# Transfer neuron linear activation
def linear(activation):
	return np.array([activation[i] for i in range(len(activation))])

# Forward propagate input to a network output
# 一次對一個個體的X P個變量
#x=x[0,]
#forward_propagate(model, x[0,])
    #x[1,:]
    #forward_propagate(model, x[1,:])
def forward_propagate(model, x):
    inputs = np.append(1,x) #每一層的截距項的INPUT都是1
    innerZsummation=[]
    inneroutput=[]
    count=1
    layer=model[0]
    layer.shape
    for layer in model:
        summation = layer.dot(np.array(inputs))
        if(count!=len(model)):
            a=sigmoid(summation)
            count+=1
        else:
            a=linear(summation)
        innerZsummation.append(summation)
        inneroutput.append(a)
        inputs = np.append(1,a) # 迴圈中每一層 hidden layer result after activity function including intercept 就是 下一層的 INPUT
    return [inputs[1:],inneroutput,innerZsummation] #inputs[1:] OUTPUT layer result after activity function
                                    #inneroutput 每一層 hidden layer result after activity function no including intercept

#因該要多加 每一層還沒經過ACTIVITIVE FUNCTION 值
# Loss function derivative
def lossfunction(y,yhat):
    sse=0
    for i in range(len(yhat)):
        sse+=(yhat[i] - y[i])**2
    mse=1/len(y)*sse
    return  (mse)
def lossfunction_derivative(y,yhat):
	return  np.array([yhat[i] - y[i] for i in range(len(yhat))] )

# Activity function derivative
def transfer_derivative(output):
	return np.array([output[i] * (1.0 - output[i]) for i in range(len(output))])
def linear_derivative(output):
	return np.array([1 for i in range(len(output))])

def corresmulti(x,y):
    return np.array([x[i]*y[i] for i in range(len(x))]).reshape(-1,1)


i=1
# Backpropagate 
def backward_propagate_error(model, y, yhat,x):
    dL_div_dyhat=lossfunction_derivative(y,yhat[0])  # yhat[0]=>OUTPUT layer after activity function
    dyhat_div_dzo=linear_derivative(yhat[0])
    end=dL_div_dyhat*dyhat_div_dzo
    end=np.array(end) #end backward_propagate to Output layer
    dL_div_weight = list()   
    temp=[]
    for i in reversed(range(len(model))):
		#layer = network[i]
        temp=[]
        if i!=0: # Output layer weight estimate and backward_propagate to k-th hidden layer
            for j in end.reshape(-1,1): #end backward_propagate to Output layer
                temp.append(np.append(1,yhat[1][i-1])*j) # K-th hidden layer output * 由後往前微分結果 (K+1)*O
            # j=1,2,...q  backward_propagate to output layer
            # yhat[1][i-1] k-1 -th hidden layer result after activity function

            dL_div_weight.append(np.array(temp)) #K-th hidden layer weight gradiant 
            #model[i].T[1:,:] 當次參數值
            dz_div_da=model[i].T[1:,:].dot(end.reshape(-1,1)) # backward_propagate to k-th hidden layer
            #model[i].T[1:,:]  btw last hidden layer and output layer weight no including intercept
            # dz_div_da for btw output layer before activity function and k-th hidden layer after activity function =
            # end backward_propagate to Output layer
      
            
            da_div_dz=transfer_derivative(yhat[1][i-1]) 
            # K-th hidden layer sigmoid activity function derivative
            # da_div_dz for btw last k-th hidden layer before activity function and k-th hidden layer after activity function

            end=corresmulti(dz_div_da , da_div_dz)
# =============================================================================
#         if (i!=0 and i!=len(model)-1):
#             for j in end.reshape(-1,1):
#                 temp.append(np.append(1,yhat[1][i-1])*j) # hidden layer output layer weight (K+1)*O
#             dL_div_weight.append(np.array(temp))
#             dz_div_da=model[i].T[1:,:].dot(end.reshape(-1,1)) # Output and K hidden layer (K+1)weight backward output layer weight*end
#             da_div_dz=transfer_derivative(yhat[1][i-1]) # K-th hidden layer a into transfer_derivative
#             end=corresmulti(dz_div_da , da_div_dz)    
# =============================================================================
        if i==0:
            for jj in end.reshape(-1,1):
                temp.append(np.append(1,x)*jj)
            # jj=1,2,...h1  backward_propagate to 1st hidden layer
            # np.append(1,x) input layer result
            dL_div_weight.append(np.array(temp))
    return(dL_div_weight)

# gradient descent method 
# batch_size=1 means SGD
# batch_size>1 <n means Mini batch GD   
# batch_size=n means GD   
    
batch_size=4 
def MBGD(x,model,y,batch_size,idx):
    total=[]
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
    batch_label=[]
    idx_culmulate=set()
    for i in range(int(len(train)/batch_size)):
        idx=np.random.choice(list(set(range(len(x)))-idx_culmulate), size=batch_size, replace=False)
        idx_culmulate.update(set(idx))
        batch_label.append(idx)

    for epoch in range(n_epoch): #每一次學將資料切分 K等分 一次epoch更新參數
        for idx in batch_label: #每一次選K等分資料中其中一份
            gradient=MBGD(x,model,y,batch_size,idx)
            for j in range(len(model)): #將每一層參數都學會
                model[j]+= -(l_rate * gradient[len(model)-1-j])  # model[j] 初始化的參數會被疊代
            # 參數總共被更新 n_epoch *(training size/batch_size) 次   
            # model[j] 順序是由1st hidden to output
            # MBGD(x,model,y,batch_size) 順序是由output to 1st hidden
            # 假設model有5層 len(model)=5 j=0,1,2,3,4 
            # j=0 ; len(model)-1-j=5-1-0=4  j=4 ; len(model)-1-j=5-1-4=0  
    #print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
    return(model)

# Make a prediction with a network
def predict(anetwork, x):
	outputs = [forward_propagate(anetwork, x[i,:])[0] for i in range(len(x))]
	return outputs 

# Generate dataset
seed(1)
mu=[5,6,7,8,9]
sigma =[5,4,3,2,1]
len(sigma)
def genernormal(mu,sigma,n):
    for i in range(len(mu)):
        if i==0:
            data=np.random.normal(mu[i], sigma[i],size=(n,1))      
        else:
            data=np.concatenate(  (data,np.random.normal(mu[i], sigma[i],size=(n,1))),axis=1 )
    return(data)
def normallization(x):
    temp=np.zeros((np.size(x,0),np.size(x,1)))
    for i in range(np.size(x,1)):
        for j in range(np.size(x,0)):
            temp[j,i]=(x[j,i]-min(x[:,i]))/(max(x[:,i])-min(x[:,i]))
    return(temp)


x= genernormal(mu,sigma,n=40)
x_n=normallization(x)
y=np.random.normal(100, 10, size=(40,1))  
y_n=normallization(y)


# User defined #input /hidden / output layer count   
model=[]
model.append(initialize_network(5, 20)[0])
#model.append(initialize_network(6, 5)[0])
#model.append(initialize_network(10, 10)[0])
#model.append(initialize_network(10, 10)[0])
#model.append(initialize_network(10, 10)[0])
model.append(initialize_network_output(20, 1)[0])
yyhat_initial=np.array(predict(model, x_n))
yyhat_initial=yyhat_initial*(max(y[:,0])-min(y[:,0]))+min(y[:,0])
sum_error_initial=lossfunction(y,yyhat_initial)
sum_error_initial
#plt.plot(yyhat_initial,y,'.')


anetwork=train_network_MBGD(model, x_n, 0.35, 5000 ,y_n,4)  # 分4個BATCH N=40故每一BATCH是10 
yyhat=np.array(predict(anetwork, x_n))

yyhat=yyhat*(max(y[:,0])-min(y[:,0]))+min(y[:,0])
sum_error=lossfunction(y,yyhat)
sum_error

import matplotlib.pyplot as plt
plt.plot(yyhat,y,'.')
plt.plot(y,y,'.')





