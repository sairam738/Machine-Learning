import pandas as pd
import numpy as np
import random
df = pd.read_csv('Train.csv')
df1 = pd.read_csv('Test.csv')
nI = int(input('enter number of input neurons '))
nO = int(input('enter number of output neurons '))
nH = int(input('enter number of Hidden neurons '))
pat = int(input('enter number of Training patterns '))
n_t = int(input('enter number of testing patterns '))
lr = float(input('enter Learning rate '))
I = np.zeros((nI,pat))
O = np.zeros((nO,pat))
I_t = np.zeros((nI,n_t))
O_t = np.zeros((nO,n_t))
aI = np.transpose(df.values[:,0:nI])
for i in range(nI):
    I[i] = 0.8*(aI[i] - min(aI[i]))/((max(aI[i]))-min(aI[i])) + 0.1
aO = np.transpose(df.values[:,nI:nI+nO])
for i in range(nO):
    O[i] = 0.8*(aO[i] - min(aO[i]))/((max(aO[i]))-min(aO[i])) + 0.1
aI_t = np.transpose(df1.values[:,0:nI])
for i in range(nI):
    I_t[i] = 0.8*(aI_t[i] - min(aI_t[i]))/((max(aI_t[i]))-min(aI_t[i])) + 0.1
aO_t = np.transpose(df1.values[:,nI:nI+nO])
for i in range(nO):
    O_t[i] = 0.8*(aO_t[i] - min(aO_t[i]))/((max(aO_t[i]))-min(aO_t[i])) + 0.1
Iones = np.ones((1,pat))
I = np.vstack([Iones,I])
I_tones = np.ones((1,n_t))
I_t = np.vstack([I_tones,I_t])
V = np.random.uniform(-1,1,(nI+1,nH+1))
W = np.random.uniform(-1,1,(nH+1,nO))
IH = np.zeros((nH+1,pat))
OH = np.zeros((nH,pat))
OH = np.vstack([Iones,OH])
IO = np.zeros((nO,pat))
OO = np.zeros((nO,pat))
E = np.zeros((nO,pat))
x = np.array(range(pat))
def ForPass(I,O,V,W,nI,nH,nO,pat):
    random.shuffle(x)
    IH = np.zeros((nH + 1, pat))
    IO = np.zeros((nO, pat))
    for p in range(pat):
        for j in range(1,nH+1):
            for i in range(nI+1):
                IH[j][p] = IH[j][p] + I[i][p]*V[i][j]
            OH[j][p] = 1/(1+np.exp(-IH[j][p]))
        for k in range(nO):
            for j in range(nH+1):
                IO[k][p] = IO[k][p] + OH[j][p]*W[j][k]
            OO[k][p] = 1 / (1 + np.exp(-IO[k][p]))
            #E[k][p] = 0.5*(OO[k][p]-O[k][p])*(OO[k][p]-O[k][p])
def BackProp(O,OO,OH,W,V,I,nH,nO,nI,pat):
    for j in range(nH+1):
        for k in range(nO):
            for p in range(pat):
                W[j][k] += (lr/pat)*((O[k][p]-OO[k][p])*OO[k][p]*(1-OO[k][p])*OH[j][p])
    for i in range(nI+1):
        for j in range(1,nH+1):
            for k in range(nO):
                for p in range(pat):
                    V[i][j] += (lr/(nO*pat))*((O[k][p]-OO[k][p])*OO[k][p]*(1-OO[k][p])*W[j][k]*OH[j][p]*(1-OH[j][p])*I[i][p])

def test(I_test,O_test,V,W,nI,nH,nO,pat):
    IH = np.zeros((nH + 1, pat))
    IO = np.zeros((nO, pat))
    for p in range(pat):
        for j in range(1, nH + 1):
            for i in range(nI + 1):
                IH[j][p] = IH[j][p] + I[i][p] * V[i][j]
            OH[j][p] = 1 / (1 + np.exp(-IH[j][p]))
        for k in range(nO):
            for j in range(nH + 1):
                IO[k][p] = IO[k][p] + OH[j][p] * W[j][k]
            OO[k][p] = 1 / (1 + np.exp(-IO[k][p]))
            OO[k][p] = (OO[k][p] - 0.1)*((max(aO[k]))-min(aO[k]))/0.8 + min(aO[k])
            E[k][p] = 0.5 * (OO[k][p] - aO[k][p]) * (OO[k][p] - aO[k][p])


for e in range(1000):

    ForPass(I,O,V,W,nI,nH,nO,pat)
    BackProp(O,OO,OH,W,V,I,nH,nO,nI,pat)

test(I_t, O_t, V, W, nI, nH, nO, n_t)
print("Mean square error for testing pattern is ",np.mean(E[0]))


