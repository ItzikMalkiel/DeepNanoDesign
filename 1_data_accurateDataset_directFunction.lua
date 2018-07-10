--load data for training the direct network that predicts the spectrums
require 'mattorch'

print '==> load train data'
trainSamples = {}
trainFiles = {}
loadDataFromPath('dataset/'..DATASET_NAME..'/train/', trainSamples, trainFiles )

if filterNoPairedData then
  trainSamples, trainFileNames = filterNoPairDirectData(trainSamples, trainFiles)
end


print '==> init train data'
columns = 1
rows = 35
labelRows = 43
trainData = torch.FloatTensor(#trainSamples, rows,columns)
trainLabels = torch.FloatTensor(#trainSamples,labelRows)
initData(trainData, trainLabels, trainSamples)

print '==> load test data'
testSamples = {}
testFiles = {}
loadDataFromPath('dataset/'..DATASET_NAME..'/test/', testSamples, testFiles )

if filterNoPairedData then
  testSamples, testFileNames = filterNoPairDirectData(testSamples, testFiles)
end
print '==> init test data'
testData = torch.FloatTensor(#testSamples, rows,columns)
testLabels = torch.FloatTensor(#testSamples,labelRows)
initData(testData, testLabels, testSamples)

trainData = trainData:transpose(2,3)
testData = testData:transpose(2,3)
numOfFeatures = trainData:size()[3]
print '==> number of features: ' 
print (numOfFeatures)


print '==> normalizing the data...'
dofile 'normalizeData.lua'
print '==> normalization is DONE!'

