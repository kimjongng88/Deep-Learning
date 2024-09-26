################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################
################################ To run this code type in one of the following: ################################
###################################### que_2(), to run question 2 ##############################################
###################################### que_3(), to run question 3 ############################################## 
################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################

### Author: Chi Him Ng
### Studentnumber: 2748786

import math
import random
import matplotlib.pyplot as plt
from statistics import mean 
plt.style.use('seaborn-darkgrid')


import numpy as np
import math


import pandas as pd

from numpy import array, half
from numpy import argmax
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

'''
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
'''
from urllib import request
import gzip
import pickle
import os



def load_synth(num_train=60_000, num_val=10_000, seed=0):#60_000 en 10_000
    """
    Load some very basic synthetic data that should be easy to classify. Two features, so that we can plot the
    decision boundary (which is an ellipse in the feature space).
    :param num_train: Number of training instances
    :param num_val: Number of test/validation instances
    :param num_features: Number of features per instance
    :return: Two tuples and an integer: (xtrain, ytrain), (xval, yval), num_cls. The first contains a matrix of training
     data with 2 features as a numpy floating point array, and the corresponding classification labels as a numpy
     integer array. The second contains the test/validation data in the same format. The last integer contains the
     number of classes (this is always 2 for this function).
    """
    np.random.seed(seed)

    THRESHOLD = 0.6
    quad = np.asarray([[1, -0.05], [1, .4]])

    ntotal = num_train + num_val

    x = np.random.randn(ntotal, 2)

    # compute the quadratic form
    q = np.einsum('bf, fk, bk -> b', x, quad, x)
    y = (q > THRESHOLD).astype(np.int)

    return (x[:num_train, :], y[:num_train]), (x[num_train:, :], y[num_train:]), 2

def load_mnist(final=False, flatten=True):
    """
    Load the MNIST data.
    :param final: If true, return the canonical test/train split. If false, split some validation data from the training
       data and keep the test data hidden.
    :param flatten: If true, each instance is flattened into a vector, so that the data is returns as a matrix with 768
        columns. If false, the data is returned as a 3-tensor preserving each image as a matrix.
    :return: Two tuples and an integer: (xtrain, ytrain), (xval, yval), num_cls. The first contains a matrix of training
     data and the corresponding classification labels as a numpy integer array. The second contains the test/validation
     data in the same format. The last integer contains the number of classes (this is always 2 for this function).
     """

    if not os.path.isfile('mnist.pkl'):
        init()

    xtrain, ytrain, xtest, ytest = load()
    xtl, xsl = xtrain.shape[0], xtest.shape[0]

    if flatten:
        xtrain = xtrain.reshape(xtl, -1)
        xtest  = xtest.reshape(xsl, -1)

    if not final: # return the flattened images
        return (xtrain[:-5000], ytrain[:-5000]), (xtrain[-5000:], ytrain[-5000:]), 10

    return (xtrain, ytrain), (xtest, ytest), 10

# Numpy-only MNIST loader. Courtesy of Hyeonseok Jung
# https://github.com/hsjeong5/MNIST-for-Numpy

filename = [
["training_images","train-images-idx3-ubyte.gz"],
["test_images","t10k-images-idx3-ubyte.gz"],
["training_labels","train-labels-idx1-ubyte.gz"],
["test_labels","t10k-labels-idx1-ubyte.gz"]
]

def download_mnist():
    base_url = "http://yann.lecun.com/exdb/mnist/"
    for name in filename:
        print("Downloading "+name[1]+"...")
        request.urlretrieve(base_url+name[1], name[1])
    print("Download complete.")

def save_mnist():
    mnist = {}
    for name in filename[:2]:
        with gzip.open(name[1], 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1,28*28)
    for name in filename[-2:]:
        with gzip.open(name[1], 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=8)
    with open("mnist.pkl", 'wb') as f:
        pickle.dump(mnist,f)
    print("Save complete.")

def init():
    download_mnist()
    save_mnist()

def load():
    with open("mnist.pkl",'rb') as f:
        mnist = pickle.load(f)
    return mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"]


################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################
################################################## QUESTION 2 ##################################################
################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################

def param():

    w1 = [[1., 1., 1.], [-1., -1., -1.]] 
    w2 = [[1., 1.], [-1., -1.], [-1., -1.]] # [[8., 1.], [-9., -1.], [-5., -1.]]
    b1 = [0.0, 0.0, 0.0]
    b2 = [0,0]
    pam_set = {"w1": w1, "b1": b1, "w2": w2, "b2": b2}
    return pam_set

def sigmoid(x):
    return 1/(1+math.exp(-x))

def sigmoid_3(x):
    return 1/(1+np.exp(-x))

def softmax_3(x):
    haha = np.exp(x)
    return haha / haha.sum(axis=1, keepdims=True)

def sig_drw_3(x):
    return sigmoid_3(x) *(1-sigmoid_3(x))

def softmax(x):
    full_sum = 0
    for i in range(len(x)):
        full_sum += math.exp(x[i])
    result = [math.exp(i)/full_sum for i in x]
    return result


def sigmoid_drw(x):
    return sigmoid(x) *(1-sigmoid (x))


def loss_function(y_pred,y):
    p = 0
    #for i in range(0,(y)):
    p += -(np.log(y_pred) * y)
    return p

def onehot(y):
    # define example
    data = y
    data = array(data)
    # one hot encode
    encoded = to_categorical(data) ##### I assume this one is allowed, altough it is not numpy. No calculation is made, just making it easier for myself.
    #print(encoded)
    return encoded

def plot_ex2(m_t_array, m_v_array):
    print(m_t_array,"adaasdfdfsds")
    plt.figure(figsize=(17,6))
    plt.subplot(121)
    plt.plot(m_t_array,label ='Training loss',color='Blue')
    plt.plot(m_v_array,label ='Validation loss',color = 'Red')
    plt.legend(fontsize=10)
    plt.ylabel('Loss',size=10)
    plt.xlabel('Epochs',size=10)
    plt.ylim(0,1)
    plt.savefig('loss.jpg', bbox_inches='tight', dpi=500)
    plt.show()

def plot_acc( test_accuracy, val_accuracy):
    plt.figure(figsize=(17,6))
    plt.subplot(121)
    plt.plot(test_accuracy,label ='Training',color='Blue')
    plt.plot(val_accuracy,label ='Validation',color = 'Red')
    plt.legend(fontsize=10)
    plt.xlabel('Epochs',size=10)
    plt.ylabel('Accuary',size=10)
    plt.ylim(0,0.7)
    plt.savefig('accuracy.jpg', bbox_inches='tight', dpi=500)
    plt.show()

def forward(x,pam):
    w1 = pam['w1']
    b1 = pam['b1']
    w2 = pam['w2']
    b2 = pam['b2']
    
    z1 = [0.,0.,0.]
    a1 = [0.,0.,0.]
    z2 = [0.,0.]
    a2 = [0.,0.]
    
    for i in range(2):
        for j in range(3):
            z1[j] += w1[i][j]*x[i]
    for j in range(3):
        z1[j] += b1[j]
        
    for i in range(3):
        a1[i] = sigmoid(z1[i])
    
    for i in range(3):
        for j in range(2):
            z2[j] += w2[i][j]*a1[i]
    for j in range(2):
        z2[j] += b2[j]
        
    a2 = softmax(z2)
        
    save_pams = {"z1": z1,"a1": a1,"z2": z2,"a2": a2}

    return a2, save_pams

def normalizedata(data):
    normalized_input = (data - np.min(data)) / (np.max(data) - np.min(data))
    return 2*normalized_input - 1

def backward(pam, saved, x, y):
    a1 = saved['a1']
    a2 = saved['a2']
    #w1 = pam['w1']
    w2 = pam['w2']
    dW = [[0.,0.,0.], [0.,0.,0.]]
    da1 = [0.,0.,0.]
    db = [0.0, 0.0, 0.0]
    dV = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
    dz1 = [0.,0.,0.]
    dc = [0.,0.]
    
    for i in range(3):#range(len(da1)):
        for j in range(2):
            dV[i][j] = a1[i]*(a2[j]-y[j])
            da1[i] += (a2[j]-y[j])*w2[i][j]
            
    for j in range(2):
        dc[j] = a2[j]-y[j]
       
    for i in range(3):
        dz1[i] = da1[i]*(a1[i]*(1-a1[i]))
        
    for i in range(2):
        for j in range(3):
            dW[i][j] = dz1[j]*x[i]
    
    for j in range(3):
        db[j] = dz1[j]   
    gradient = {"dW": dW,"db": db,"dV": dV,"dc": dc}

    return gradient

def update_pams(pam, gradient,l_rate):
    
    dW = gradient['dW']
    db = gradient['db']
    dV = gradient['dV']
    dc = gradient['dc']
    w1 = pam['w1']
    b1 = pam['b1']
    w2 = pam['w2']
    b2 = pam['b2']
    
    for i in range(2):
        for j in range(3):
            w1[i][j] = w1[i][j] - l_rate*dW[i][j]
            
    for i in range(3):
        b1[i] = b1[i] - l_rate*db[i]
    
    for i in range(3):
        for j in range(2):
            w2[i][j] = w2[i][j] - l_rate*dV[i][j]
    for i in range(2):
        b2[i] = b2[i] - l_rate*dc[i]
        
    pam = {"w1": w1,"b1": b1,"w2": w2,"b2": b2}
    return pam
    
def forward_backward(xtrain,ytrain,xval,y_val,l_rate,iterations):
    pam = param()
    loss = []
    val_loss = []
    train_acc_list = []
    val_acc_list = []
    for epoch in range(iterations):
        t_acc = 0
        v_acc = 0
        
        #Forward propagation question 2, this is used to check the loss (cross-entropy in our case)
        for k in range(len(xval)):
            forward_out, _ = forward(xval[k],pam)
            ce_loss = loss_function(forward_out,y_val[k])
            val_loss.append(ce_loss)
            
        # And then we go forward with x training dataset, then immeadiatly backwards    
        for i in range(len(xtrain)):
            a2, save = forward(xtrain[i],pam)
            ll = loss_function(a2,ytrain[i])
            #print(ll,"hierlopenwevast")
            loss.append(ll)
            gradient = backward(pam, save, xtrain[i], ytrain[i]) # Also get the gradients
            pam = update_pams(pam, gradient,l_rate) # Update parameters with the outcomes

        # Training accuracy
        for i in range(len(xtrain)):
            a2, save = forward(xtrain[i],pam)
            max_list = a2.index(max(a2))
            if ytrain[i][max_list] == 1:
                t_acc+=1
        train_acc_list.append(t_acc/len(ytrain))

        # Validation, now we go forward again and compare the output
        for i in range(len(xval)):
            a2, save = forward(xval[i],pam)
            max_list = a2.index(max(a2))
            if y_val[i][max_list] == 1:
                v_acc+=1
        val_acc_list.append(v_acc/len(y_val))

    
    return loss, val_loss,train_acc_list,val_acc_list,pam


def que_2():
    (xtrain, ytrain), (xval, yval), num_cls = load_synth()

    ytrain = onehot(ytrain)
    yval = onehot(yval)

    iterations = 1
    l_rate=0.005

    loss, val_loss,test_accuracy,val_accuracy, pampam = forward_backward(xtrain,ytrain,xval, yval,l_rate,iterations)

    print(pampam)

    t_array = []
    v_array = []
    m_t_array = []
    m_v_array = []
    for j in range(0,iterations):
        for i in range(j*len(xtrain), (j + 1)*len(xtrain)):#100
            t_array.append(sum(loss[i])/2)
        for a in range(j*len(xval), (j+1) * len(xval)):
            v_array.append(sum(val_loss[a])/2)
        m_t_array.append(np.mean(t_array))
        t_array = []
        m_v_array.append(np.mean(v_array))
        v_array = []
        
        

    #plot_ex2(m_t_array, m_v_array)

    #plot_acc(test_accuracy, val_accuracy)

    print(m_t_array,'\n',m_v_array,"herman the german")


################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################
################################################## QUESTION 3 ##################################################
################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################

def forward_prop_p3(x,pam):
    #print(x, "x")
    z1 = np.dot(x, pam['W']) + pam['b']
    #print(z1,'not iterable correct')
    a1 = sigmoid_3(z1)
    #print(a1, "dit is a1")


    z2 = np.dot(a1, pam['V']) + pam['c']
    #print(z2, "dit is z2")
    a2 = softmax_3(z2)
    #print(a2, "wij zijn bijna klaar")
    
    ##save data
    save = {"z1": z1,"a1": a1,"z2": z2,"a2": a2}
    return a2, save
    
def backward_prop_p3(pam, save,x, y):
    ## Softmax part
    loss_dos = save['a2'] - y
    pp = save['a1']
    #print(pp)
    loss_ppT_ld = np.dot(pp.T, loss_dos)
    dloss_c = loss_dos

    ## Sigmoid and co
    dloss_da1 = np.dot(loss_dos,pam['V'].T) #np.dot(dz2_da1.T,loss_dos)
    da1_dz1 = sig_drw_3(save['z1'])
    #print(dz2_da1,"dz2", dloss_dal,"dloss", da1_dz1,"da1", x,'laatste)
    loss_dwT = np.dot(x.T, da1_dz1 * dloss_da1)
    l_loss = dloss_da1 * da1_dz1
    
    gradient = {"loss_dwT": loss_dwT,"l_loss": l_loss,"loss_ppT_ld": loss_ppT_ld,"dloss_c": dloss_c}
    return gradient


def predictionmaker(pam,x,true):
    #Checking correctness
    ac = 0
    b, nadahihihi = forward_prop_p3(x,pam)
    j = np.argmax(b,axis=1)
    for i in range(len(true)):
        if true[i][j[i]] == 1:
            ac +=1
    ac = ac/len(b)
    return ac


def c_loss(arr,y,pre):
    return np.sum(-y * np.log(arr))/pre
    

def update_pam(pam,gradient,learn_rate):
    W = pam['W']
    b = pam['b']
    V = pam['V']
    c = pam['c']
    W = W - learn_rate * gradient["loss_dwT"] # + momentum * W
    b = b - learn_rate * gradient["l_loss"].sum(axis=0)
    V = V - learn_rate * gradient["loss_ppT_ld"] # + momentum * V
    c = c-  learn_rate * gradient["dloss_c"].sum(axis=0)
    
    pam = {"W": W,"b": b,"V": V,"c": c}
    return pam

def model_p3(xtrain,ytrain,xval,y_val,learn_rate,batchsize,pam,m, iterations):
    total_loss =[]
    val_loss = []
    training_accuracy =[]
    vali_accu =[]
    for jjj in range(iterations):#TODO change loop if necessary
        print(jjj)
        
        #Per batch
        if m == 'batch':
            train_shuffling = np.arange(0,len(xtrain))
            random.shuffle(train_shuffling)

            lalala = np.arange(0,len(xval))            
            random.shuffle(lalala) 
            
            for i in range(int(len(xtrain)/batchsize)): #range(test_len):
                resultx = xtrain[train_shuffling[i*batchsize:(i+1)*batchsize]]
                resulty = ytrain[train_shuffling[i*batchsize:(i+1)*batchsize]]
                
                arr, s_pam = forward_prop_p3(resultx,pam)
                loss = c_loss(arr,resulty,int(len(xtrain)/batchsize))
                total_loss.append(loss)
                gradient = backward_prop_p3(pam, s_pam,resultx, resulty)
                pam = update_pam(pam,gradient,learn_rate)
                
            for i in range(int(len(xval)/batchsize)):
                bambam,_ = forward_prop_p3(xval[lalala[i*batchsize:(i+1)*batchsize]],pam)
                v_loss = c_loss(bambam,y_val[lalala[i*batchsize:(i+1)*batchsize]],int(len(xval)/batchsize))
                val_loss.append(v_loss)             
            train_acc = predictionmaker(pam,xtrain,ytrain)
            val_acc = predictionmaker(pam,xval,y_val)
            training_accuracy.append(train_acc)
            vali_accu.append(val_acc)
        else:
            for i in range(len(xtrain)):
                number = 1
                resultx = xtrain[i].reshape((1,784))
                y_label = ytrain[i].reshape((1,10))
                r_f_x, rfx_pam = forward_prop_p3(resultx,pam)
                loss = c_loss(r_f_x,y_label,number)
                
                total_loss.append(loss)
                gradient = backward_prop_p3(pam, rfx_pam,resultx, y_label)
                pam = update_pam(pam,gradient,learn_rate)
            for i in range(len(xval)):
                number = 1
                xval_s = xval[i].reshape((1,784))
        
                y_val_s = y_val[i].reshape((1,10))
                
                r_f, rfpam_nada = forward_prop_p3(xval_s,pam)
                v_loss = c_loss(r_f,y_val_s,number)
                val_loss.append(v_loss)
            train_acc = predictionmaker(pam,xtrain,ytrain)
            val_acc = predictionmaker(pam,xval,y_val)
            training_accuracy.append(train_acc)
            vali_accu.append(val_acc)
                
                

    return total_loss,val_loss,training_accuracy,vali_accu
            
    
def mean_batch(train_l,val_l,iterations,x,y,b):
    mean_train = []
    mean_val = []
    x = int(x/b)
    y = int(y/b)
    print(x,y)
    for i in range(iterations):
        a = np.sum(train_l[i*x:(i+1)*x])/55
        mean_train.append(a)
        mean_val.append(np.mean(val_l[i*y:(i+1)*y]))
    return mean_train,mean_val

def mean_sto(train_l,val_l,iterations,x,y):
    mean_train = []
    mean_val = []
    print(x,y)
    for i in range(iterations):
        mean_train.append(np.mean(train_l[i*x:(i+1)*x]))
        mean_val.append(np.mean(val_l[i*y:(i+1)*y]))
    return mean_train,mean_val


def final_prediction(xtrain,ytrain,xval,y_val,learn_rate,number,batchsize):
    input_nodes = 784
    hidden_nodes = 300
    output_nodes = 10


    W = np.random.normal(size=(input_nodes,hidden_nodes))
    b = np.zeros(hidden_nodes)
    V = np.random.normal(size=(hidden_nodes,output_nodes))
    c = np.zeros(output_nodes)
    
    pam = {"W": W,"b": b,"V": V,"c": c}

    for epoch in range(30):
        ############# forward
        for i in range(batchsize):
            train_shuffling =np.random.randint(len(xtrain), size=number)
            X_train = xtrain[train_shuffling]
            y_label = ytrain[train_shuffling]

            a2,save = forward_prop_p3(X_train,pam)
            loss = c_loss(a2,y_label,number=number)

            gradient = backward_prop_p3(pam, save,X_train, y_label)
            pam = update_pam(pam,gradient,learn_rate)
            
    accuracy = predictionmaker(pam,xval,y_val)
    
    return accuracy 

def plt_p3(train_l, val_l, mtrain, mval, tacc, vacc):
    plt.figure(figsize=(20,4))
    plt.subplot(131)
    plt.plot(train_l,label ='Training',color='#DC143C')
    plt.plot(val_l,alpha =0.5,label ='Validation',color = '#1E90FF')
    plt.xlabel('Steps',size=15)
    plt.ylabel('Loss',size=15)
    plt.xlim(0,55000)
    plt.legend(fontsize=15)
    plt.subplot(132)
    plt.plot(mtrain,label ='Training',color='#DC143C')
    plt.plot(mval,label ='Validation',color = '#1E90FF')
    plt.legend(fontsize=15)
    plt.ylabel('Loss',size=15)
    plt.xlabel('Epochs',size=15)
    plt.subplot(133)
    plt.plot(tacc,label ='Training',color='#DC143C')
    plt.plot(vacc,label ='Validation',color = '#1E90FF')
    plt.legend(fontsize=15)
    plt.ylabel('Accuary',size=15)
    plt.xlabel('Epochs',size=15)
    plt.savefig('results_p3__3.jpg', bbox_inches='tight', dpi=500)
    plt.show()

def plt_dom(all_array,labels,batches):
    plt.figure(figsize=(15,5))
    plt.subplot(1,2,1)
    for index, i in enumerate(all_array):
        plt.plot((np.sum(i[:,0])/batches),label='Lr = '+labels[index])
        plt.fill_between(range(len(i[:,0][0])),((np.sum(i[:,0])/batches)+(np.sum((np.std((np.sum(i[:,0]))/batches)))/batches)),((np.sum(i[:,0])/batches)-(np.sum((np.std((np.sum(i[:,0]))/batches)))/batches)),alpha=0.3)
        #plt.fill_between(range(len(i[:,0][0])),(np.mean(i[:,0])+np.std(i[:,0])),(np.mean(i[:,0])-np.std(i[:,0])),alpha=0.3)
    plt.xlabel('Batches',size=15)
    plt.ylabel('Average training loss',size=15)
    plt.legend(fontsize=13)
    plt.ylim(0,1)
    plt.savefig('lossssss_acc.jpg', bbox_inches='tight', dpi=500)
    plt.show()

def plt_dom2(all_array, labels):
    plt.figure(figsize=(15,5))
    plt.subplot(1,2,1)
    for index, i in enumerate(all_array): 
        plt.plot(np.mean(i[:,1]),label='Lr = '+labels[index])
        plt.fill_between(range(len(i[:,1][0])),(np.mean(i[:,1])+np.std(i[:,1])),(np.mean(i[:,1])-np.std(i[:,1])),alpha=0.3)
    plt.xlabel('Epochs',size=15)
    plt.ylabel('Training accuracy',size=15)
    plt.legend(fontsize=13)
    # plt.ylim(0,15)
    plt.savefig('training_acc.jpg', bbox_inches='tight', dpi=500)
    plt.show()


def que_3():
    (x, y), (xtest, ytest), a = load_mnist()

    # 'batch' if you want to use batch, fill in any string if not batch
    method = 'batch'



    input_nodes = 784
    hidden_nodes = 300
    output_nodes = 10
    momentum = 0.9
    iterations = 20
    l_rate = 0.01

    x = normalizedata(x)
    y = onehot(y)
    xtest = normalizedata(xtest)
    ytest = onehot(ytest)

    W = np.random.normal(size=(input_nodes,hidden_nodes))
    b = np.zeros(hidden_nodes)
    V = np.random.normal(size=(hidden_nodes,output_nodes))
    c = np.zeros(output_nodes)
    
    pam = {"W": W,"b": b,"V": V,"c": c}
    batchsize = int(len(x)/1000)
    print(batchsize)

    '''
    l_rate = 0.001
    l_01 = []
    for i in range(5):
        train_loss, _, mini_t_a,_ = model_p3(x,y,xtest,ytest,l_rate,batchsize,pam,method,iterations)
        l_01.append([np.array(train_loss),np.array(mini_t_a)])
    l_01 = np.array(l_01)

    l_rate = 0.003
    l_03 = []
    for i in range(5):
        train_loss, _, mini_t_a,_ = model_p3(x,y,xtest,ytest,l_rate,batchsize,pam,method,iterations)
        l_03.append([np.array(train_loss),np.array(mini_t_a)])
    l_03 = np.array(l_03)


    l_rate = 0.01
    l_1 = []
    for i in range(5):
        train_loss, _, mini_t_a,_ = model_p3(x,y,xtest,ytest,l_rate,batchsize,pam,method,iterations)
        l_1.append([np.array(train_loss),np.array(mini_t_a)])
    l_1 = np.array(l_1)


    l_rate = 0.03
    l_3 = []
    for i in range(5):
        train_loss, _, mini_t_a,_ = model_p3(x,y,xtest,ytest,l_rate,batchsize,pam,method,iterations)
        l_3.append([np.array(train_loss),np.array(mini_t_a)])
    l_3 = np.array(l_3)

    all_array = [l_01, l_03, l_1, l_3]
    labels= ['0.001','0.003','0.01','0.03']


    plt_dom(all_array,labels,batchsize)
    plt_dom2(all_array,labels)
    '''

    train_loss, val_loss,tacc,vacc = model_p3(x,y,xtest,ytest,l_rate,batchsize,pam,method,iterations)

    #print(train_loss,val_loss,"dit is loss")

    mean_train,mean_val = mean_batch(train_loss,val_loss,iterations,len(x),len(ytest),batchsize)

    #mean_train,mean_val = mean_sto(train_loss,val_loss,iterations,len(x),len(ytest))

    #print(mean_train, mean_val, 'ole at the wheel')

    plt_p3(train_loss, val_loss, mean_train, mean_val, tacc, vacc)

que_3()
