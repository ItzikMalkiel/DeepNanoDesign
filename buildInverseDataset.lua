-----------------------------------
-- a stand alone code that reads a datset, process the data and build the inverse data
-- for the inverse functino. Then it writes the output to the disc in order to save processsing time 
--  
-- can be used by configuring the "useDatasetHook" flag as True
-----------------------------------
require 'mattorch'
require 'torch'
dofile 'utils.lua'
DATASET_NAME = 'datasetVersion4.3'          

function initData(data, labels, samples)
  print (samples[1].output_data)

  for i = 1,#samples do

    y = samples[i].output_data
    experimentVariables = y[{{1,35}}]
    waveLengths =  y[{{36,78}}]
    transmittions = y[{{79,121}}]
    R = y[{{122,164}}]

    data[{{i},{}}] = torch.cat(experimentVariables [{{9,35}}],transmittions,1)
    labels[{{i},{}}] = experimentVariables [{{1,8}}]
  end
end


print '==> load train data'
trainSamples = {}
trainFiles = {}
loadDataFromPath('dataset/'..DATASET_NAME..'/train/', trainSamples, trainFiles )



print '==> init train data'
columns = 1
rows = 43+43+35-2-8
labelRows = 8
trainData = torch.FloatTensor(10000, rows,columns)--version 4.1: 4819. versionn 4.3: 2896 . inverse test 965
trainLabels = torch.FloatTensor(10000,labelRows)--version 4.1: among 9700. version 4.1:  among 5849 
trainFiles= mergeExperimentsByPolarization(trainSamples, trainFiles,trainData, trainLabels)
print ("new experiments names size is: " .. #trainFiles)
wait(5)
--initData(trainData, trainLabels, trainSamples)

print '==> load test data'
testSamples = {}
testFiles = {}
loadDataFromPath('dataset/'..DATASET_NAME..'/test/', testSamples, testFiles )


print '==> init test data'
testData = torch.FloatTensor(1250, rows,columns)--1791.   versionn 4.3: 1250. inverse test 425
testLabels = torch.FloatTensor(1250,labelRows)-- among  3583. versionn 4.3: 2501  among ~850
testFiles = mergeExperimentsByPolarization(testSamples, testFiles,testData, testLabels)
print ("new experiments names size is: " .. #testFiles)
wait(5)


trainData = trainData:transpose(2,3)
testData = testData:transpose(2,3)
numOfFeatures = trainData:size()[3]
print '==> how many features we have? good question! the answer is: ' 
print (numOfFeatures)


--write data to disk
torch.save('inverseDataset/trainData', trainData)
torch.save('inverseDataset/testData', testData)
torch.save('inverseDataset/trainLabels', trainLabels)
torch.save('inverseDataset/testLabels', testLabels)
