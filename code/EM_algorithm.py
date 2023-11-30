import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from tqdm import tqdm

# Read data
data = pd.read_csv('data.csv')
train_data = data['Height']

# Step 1: Initialize parameters
mu1 = 170.
mu2 = 170.
sigma1 = 10.
sigma2 = 10.
p = 0.6
L = list()

def cal_loss(mu1, mu2, sigma1, sigma2, p):
    loss = 0
    for i in range(len(train_data)):
        loss += (-np.log(p*stats.norm(mu1, sigma1).pdf(train_data[i]) + (1.-p)*stats.norm(mu2, sigma2).pdf(train_data[i])))
    return loss

gauss1 = pd.Series(np.zeros(len(train_data)))
def EM(gauss1, mu1, mu2, sigma1, sigma2, p):
    loss = cal_loss(mu1, mu2, sigma1, sigma2, p)
    L.append(loss)
    # Step 2: E-step
    for i in range(len(train_data)):
        gauss1[i] = p*stats.norm(mu1, sigma1).pdf(train_data[i]) 
        gauss1[i] /= (p*stats.norm(mu1, sigma1).pdf(train_data[i]) + (1.-p)*stats.norm(mu2, sigma2).pdf(train_data[i]))
    
    # Step 3: M-step
    sigma1 = np.sqrt(np.sum(gauss1*(train_data-mu1)**2.)/np.sum(gauss1))
    mu1 = np.sum(gauss1*train_data)/np.sum(gauss1)
    sigma2 = np.sqrt(np.sum((1.-gauss1)*(train_data-mu2)**2.)/np.sum(1.-gauss1))
    mu2 = np.sum((1.-gauss1)*train_data)/np.sum(1.-gauss1)
    # print(np.sum(gauss1))
    p = np.sum(gauss1)/len(gauss1)

    return mu1, mu2, sigma1, sigma2, p

# Step 4: Iterate E-step and M-step
for i in range(11):
    mu1, mu2, sigma1, sigma2, p = EM(gauss1, mu1, mu2, sigma1, sigma2, p)
    print(mu1, mu2, sigma1, sigma2, p, i)
    
# Step 5: Plot the loss curve
# x = np.arange(200)
# Loss = np.array(L) / len(train_data)
# plt.plot(x, Loss, color='blue', linewidth=3)
# plt.xlabel('Iteration')
# plt.ylabel('Loss')
# plt.title('Loss Curve')
# plt.savefig('LossCurve.png', dpi=300)