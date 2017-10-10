import sys
import numpy as np
import math
import matplotlib.pyplot as plt

def generateSamples(n,m):
    x = np.random.rand(n,m)
    h = 0.5 + (0.4 * np.sin(np.pi*2*x))
    output = h + np.random.uniform(-0.1,0.1,(n,m))
    return [x,h,output]

def getInitialPoints(data,k):
    index = np.random.randint(0,data.shape[0],k)
    centroids = data[index]
    return centroids

def euclidianDistance(row1, row2):
    difference = row1-row2
    dist = math.sqrt(np.dot(difference,difference.transpose()))
    return dist

def assignClusters(data,centroids,k):
    clusterNum=  np.zeros((data.shape[0]), dtype=np.int)
    for i in range(0, data.shape[0]):
        clusterDist = -1
        for j in range(0,k):
            dist = euclidianDistance(data[i], centroids[j])
            if(dist<clusterDist or clusterDist==-1):
                clusterDist = dist
                clusterNum[i] = j
    return clusterNum

def findFarthestPoints(centroids,data,clusterClass,k):
    newPoints= int(k - len(np.unique(clusterClass)))
    datapointDistances =  np.zeros((data.shape[0]), dtype=[('dist',float),('id',int)])
    for i in range(0, data.shape[0]):
        clusterVal = clusterClass[i]-1
        datapointDistances[i][0] = euclidianDistance(data[i], centroids[clusterVal])
        datapointDistances[i][1] = i
    datapointDistances = np.sort(datapointDistances,order='dist')[::-1]
    return datapointDistances[0:newPoints]

def updateCentroids(data,clusterClass,k):
    newCentroids = np.zeros((k,data.shape[1]), dtype=np.float)
    flag = False
    emptyClusters = []
    for j in range(0,k):
        totalCount = 0
        for i in range(0, data.shape[0]):
            if(clusterClass[i]==j):
                newCentroids[j] = newCentroids[j] + data[i,:]
                totalCount = totalCount + 1
        if(totalCount!=0):
            newCentroids[j] = newCentroids[j] / totalCount
        # clusters with no data points
        else:
            flag = True
            emptyClusters.append(j)
    if(flag):
        centroidsForEmptyCluster = findFarthestPoints(newCentroids,data,clusterClass,k)
        for val in range(0, len(emptyClusters)):
            index = centroidsForEmptyCluster[val][1]
            newCentroids[emptyClusters[val]] = data[index]
    return newCentroids

def kMeansCluster(data,k):
    centroids = getInitialPoints(data,k)
    check = True
    counter = 1
    while(check):
        clusterClass = assignClusters(data,centroids,k)
        newCentroids = updateCentroids(data,clusterClass,k)
        counter+=1
        if((newCentroids==centroids).all()):
            check = False
        else:
            centroids = newCentroids
    # print (counter,centroids,clusterClass)
    return [centroids,clusterClass]

def maxDistanceInCentroid(centroids):
    maxDist = 0
    centroidSize = centroids.shape[0]
    for i in range(0, centroidSize):
        for j in range(i+1,centroidSize):
            dist = euclidianDistance(centroids[i], centroids[j])
            if(dist > maxDist):
                maxDist = dist
    return maxDist

def assignSimpleVariance(centroids,k):
    maxDist = maxDistanceInCentroid(centroids)
    variance = np.ones(k, dtype=np.float) * ((maxDist*maxDist)/(2*k))
    return variance

def assignVariance(data,centroids,clusterClass,k):
    variance = np.zeros(k, dtype=np.float)
    totalCount = np.zeros(k, dtype=np.float)
    singleClusters = []
    for i in range(0, data.shape[0]):
        clusterVal = clusterClass[i]
        dist = euclidianDistance(data[i], centroids[clusterVal])
        variance[clusterVal] = variance[clusterVal] + (dist*dist)
        totalCount[clusterVal] = totalCount[clusterVal] + 1
    for j in range(0,k):
        if(totalCount[j]>1):
            variance[j] = variance[j] / totalCount[j]
        else:
            singleClusters.append(j)
    averageVariance = (np.sum(variance))/ k
    for val in range(0, len(singleClusters)):
        variance[singleClusters[val]] = averageVariance
    return variance

def initWeights(rows,cols):
    return np.random.uniform(-1,1,(rows,cols))

def gaussian(data,mean,var):
    gauss = np.exp(-np.power((data-mean),2)/(2*var))
    return gauss

def getHiddenLayerOutput(data,centroids,variance,k):
    # Last column holds the bias inputs
    hiddenLayerOutput =  np.ones((data.shape[0],k+1), dtype=np.float)
    for i in range(0,k):
        hiddenLayerOutput[:,i] = gaussian(data,centroids[i],variance[i]).reshape(data.shape[0])
    return hiddenLayerOutput

def updateWeights(weights,hiddenLayerOutput,desiredOutput,eta,epoch,k):
    for i in range(0,epoch):
        for j in range(0,hiddenLayerOutput.shape[0]):
            F = np.dot(hiddenLayerOutput[j],weights)
            err = (desiredOutput[j] - F)
            delW = (eta * err * hiddenLayerOutput[j])
            weights = weights + delW.reshape(k+1,1)
    return weights

if __name__ == "__main__":
    sampleSize = 75
    inputLayerSize = 1
    outputLayerSize = 1
    eta1 = 0.01
    eta2 = 0.02
    epoch = 100
    [sampleInput,functionOutput,desiredOutput] = generateSamples(sampleSize,inputLayerSize)
    K = [2,4,7,11,16]
    for k in K:
        [centroids,clusterClass] = kMeansCluster(sampleInput, k)
        weights = initWeights(k+1,outputLayerSize)

        variance1 = assignVariance(sampleInput,centroids,clusterClass,k)
        hiddenLayerOutput = getHiddenLayerOutput(sampleInput,centroids,variance1,k)

        weights1 = updateWeights(weights,hiddenLayerOutput,desiredOutput,eta1,epoch,k)
        actualOutput11 = np.dot(hiddenLayerOutput,weights1)
        weights2 = updateWeights(weights,hiddenLayerOutput,desiredOutput,eta2,epoch,k)
        actualOutput12 = np.dot(hiddenLayerOutput,weights2)

        variance2 = assignSimpleVariance(centroids,k)
        hiddenLayerOutput = getHiddenLayerOutput(sampleInput,centroids,variance2,k)

        weights1 = updateWeights(weights,hiddenLayerOutput,desiredOutput,eta1,epoch,k)
        actualOutput21 = np.dot(hiddenLayerOutput,weights1)
        weights2 = updateWeights(weights,hiddenLayerOutput,desiredOutput,eta2,epoch,k)
        actualOutput22 = np.dot(hiddenLayerOutput,weights2)

        result = np.concatenate((sampleInput, functionOutput, desiredOutput, actualOutput11, actualOutput12, actualOutput21, actualOutput22), axis=1)
        result = result[result[:,0].argsort()]

        fig = plt.figure()
        plt.title('Number of bases: %i Normal Variance' % k)
        fo, = plt.plot(result[:,0],result[:,1],label='Original Function Output')
        ns = plt.scatter(result[:,0],result[:,2],alpha=0.4,label='Noisy Sample Output')
        rbf1, = plt.plot(result[:,0],result[:,3],label='RBF Net Output, learningRate: %1.2f'%eta1)
        rbf2, = plt.plot(result[:,0],result[:,4],label='RBF Net Output, learningRate: %1.2f'%eta2)
        plt.ylabel('Output')
        plt.xlabel('input (x)')
        plt.legend(handles=[fo,ns,rbf1,rbf2])
        fig.savefig('results/RBF_k=%i_1.png'%k)
        plt.show()

        fig = plt.figure()
        fo, = plt.plot(result[:,0],result[:,1],label='Original Function Output')
        plt.title('Number of bases: %i Simple Variance' % k)
        ns = plt.scatter(result[:,0],result[:,2],alpha=0.4,label='Noisy Sample Output')
        rbf1, = plt.plot(result[:,0],result[:,5],label='RBF Net Output, learningRate: %1.2f'%eta1)
        rbf2, = plt.plot(result[:,0],result[:,6],label='RBF Net Output, learningRate: %1.2f'%eta2)
        plt.ylabel('Output')
        plt.xlabel('input (x)')
        plt.legend(handles=[fo,ns,rbf1,rbf2])
        fig.savefig('results/RBF_k=%i_2.png'%k)
        plt.show()
