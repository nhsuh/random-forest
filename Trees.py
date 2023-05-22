#!/usr/bin/env python
# coding: utf-8

# In[261]:


import numpy as np
from scipy import stats


# In[262]:


train = np.loadtxt('zip_train.csv', delimiter=',')
test = np.loadtxt('zip_test.csv', delimiter = ',')


# In[263]:


ones = train[:, 0] == 1
threes = train[:, 0] == 3
fives = train[:, 0] == 5
ones_or_threes = ones | threes
threes_or_fives = threes | fives
train_OT = train[ones_or_threes]
train_TF = train[threes_or_fives]


# In[264]:


X_train_OT = train_OT[:, 1:]
print(X_train_OT.shape)
y_train_OT = train_OT[:, 0]
print(y_train_OT)
X_train_TF = train_TF[:, 1:]
y_train_TF = train_TF[:, 0]


# In[265]:


ones = test[:, 0] == 1
threes = test[:, 0] == 3
fives = test[:, 0] == 5
ones_or_threes = ones | threes
threes_or_fives = threes | fives
test_OT = test[ones_or_threes]
test_TF = test[threes_or_fives]


# In[266]:


X_test_OT = test_OT[:, 1:]
y_test_OT = test_OT[:, 0]
X_test_TF = test_TF[:, 1:]
y_test_TF = test_TF[:, 0]


# In[267]:


from sklearn.tree import DecisionTreeClassifier
from scipy import stats

def random_forest(X_train, y_train, X_test, y_test, num_bags, m):
    #get dimensiosn of X_train to get n (# of data points), p (# of dimesnsions)
    n, p = X_train.shape
    n_test, p_test = X_test.shape
    
    #initialize arrays to hold the out of bag predictions and test predictions
    oob_pred_arr = np.zeros((n, num_bags))
    test_pred_arr = np.zeros((n_test, num_bags))
    
    #train num_bags trees, then test it's ability to predict in bag and oob
    for i in range(num_bags):
        #randomly sample to get in bag
        ib_indices = np.random.choice(n, 200)
        
        #oob is anything not in bag, so set difference
        oob_indices = np.setdiff1d(np.arange(n), ib_indices)
        
        #fit a decision tree to the in bag data
        tree = DecisionTreeClassifier(max_features=m, criterion="entropy")
        tree.fit(X_train[ib_indices], y_train[ib_indices])
        
        #predict oob data for each bag
        oob_pred = tree.predict(X_train[oob_indices])
        oob_pred_arr[oob_indices, i] = oob_pred
        
        #predict test data for each bag
        test_pred = tree.predict(X_test)
        test_pred_arr[:, i] = test_pred
    
    oob_pred_maj = np.zeros(n)
    
    #get the mode of the oob predictions and get rid of 0s
    for row in range(0, len(oob_pred_arr)):
        curr_row = oob_pred_arr[row]
        curr_row = curr_row[curr_row != 0]
        if(len(curr_row) == 0):
            oob_pred_maj[row] = 0
        else:
            oob_pred_maj[row] = stats.mode(curr_row).mode
            
    #get the mode of the test predictions
    test_pred_maj = stats.mode(test_pred_arr, axis=1).mode.flatten()
    
    #binary error
    out_of_bag_error = np.mean(oob_pred_maj != y_train)
    test_error = np.mean(test_pred_maj != y_test)
    return out_of_bag_error, test_error


# In[268]:


random_forest(X_train_OT, y_train_OT, X_test_OT, y_test_OT, 50, 20)


# In[269]:


random_forest(X_train_TF, y_train_TF, X_test_TF, y_test_TF, 50, 20)


# In[291]:


trials = [random_forest(X_train_OT, y_train_OT, X_test_OT, y_test_OT, 50, 10) for _ in range(0,100)]
oobe = trials[0]
te = trials[1]


# In[ ]:





# In[292]:


oobe_min = np.min(oobe)
oobe_max = np.max(oobe)
te_min = np.min(te)
te_max = np.max(te)
print("oobe min", oobe_min)
print("oobe max", oobe_max)
print("test error min", te_min)
print("test error max", te_max)
print("oobe range", oobe_max-oobe_min)
print("test error range", te_max-te_min)


# In[293]:


trials = [random_forest(X_train_TF, y_train_TF, X_test_TF, y_test_TF, 50, 10) for _ in range(0,100)]
oobe = trials[0]
te = trials[1]


# In[294]:


oobe_min = np.min(oobe)
oobe_max = np.max(oobe)
te_min = np.min(te)
te_max = np.max(te)
print("oobe min", oobe_min)
print("oobe max", oobe_max)
print("test error min", te_min)
print("test error max", te_max)
print("oobe range", oobe_max-oobe_min)
print("test error range", te_max-te_min)


# In[295]:


trialsOT = [random_forest(X_train_OT, y_train_OT, X_test_OT, y_test_OT, b, 100)[0] for b in range(2,50)]
trialsTF = [random_forest(X_train_TF, y_train_TF, X_test_TF, y_test_TF, b, 100)[0] for b in range(2,50)]
print(trialsOT[0])
print(trialsTF[0])


# In[300]:


import matplotlib.pyplot as plt

plt.plot(trialsOT, label='1vs3')
plt.plot(trialsTF, label='3vs5')
plt.legend()
plt.title('Out of Bag Error as a function of number of bags for 1vs3 and 3vs5 problems')
plt.xlabel('Number of Bags')
plt.ylabel('Out of Bag Error')

plt.show()


# In[297]:





# In[298]:





# In[ ]:





# In[ ]:




