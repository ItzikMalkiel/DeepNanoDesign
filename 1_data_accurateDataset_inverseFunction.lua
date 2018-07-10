

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


if (loadDatasetHook) then

  --read data from disk
  trainData = torch.load('inverseDataset/trainData')
  testData = torch.load('inverseDataset/testData')
  trainLabels = torch.load('inverseDataset/trainLabels')
  testLabels =  torch.load('inverseDataset/testLabels')

  trainData = trainData:transpose(2,3)
  testData = testData:transpose(2,3)
  
  numOfFeatures = trainData:size()[3]
  print('num of features: ' .. numOfFeatures)

  if useNormalization then
    print '==> normalize Data!!'
    dofile 'normalizeData.lua'
    print '==> normalization is DONE!!'
  end
  
  if dummyData then
  
    print '==> dummyData!!'
    trainData = trainData * 0
    testData = testData * 0 
    trainLabels = trainLabels * 0 
    testLabels = testLabels * 0  
  end

else

  print '==> load train data'
  trainSamples = {}
  trainFiles = {}
  loadDataFromPath('dataset/'..DATASET_NAME..'/train/', trainSamples, trainFiles )

  print '==> init train data'
  columns = 1
  rows = 43+43+35-2-8
  labelRows = 8
  trainData = torch.FloatTensor(10000, rows,columns)--version 4.1: 4819. versionn 4.3: 2924 . inverse test 965. version 5.3:4007. datasetVersion5.4: 5363. version 5.5: 5494. after a found-noduplication fix: 5462
  trainLabels = torch.FloatTensor(10000,labelRows)--version 4.1: among 9700. version 4.1:  among 5849 
  trainData, trainLabels, trainFiles= mergeExperimentsByPolarization(trainSamples, trainFiles,trainData, trainLabels)
  print("trainData:size()")
  print(trainData:size())
  print("trainLabels:size()")
  print(trainLabels:size())
  print ("new experiments names size is: " .. #trainFiles)

  print '==> load test data'
  testSamples = {}
  testFiles = {}
  loadDataFromPath('dataset/'..DATASET_NAME..'/test/', testSamples, testFiles )

  print '==> init test data'
  testData = torch.FloatTensor(10000, rows,columns)--1791.   versionn 4.3: 1250. inverse test 425. version 5.3: 526. datasetVersion5.4:440. version 5.5: 309. version5.5Hooked:445 after a found-noduplication fix: 341
  testLabels = torch.FloatTensor(10000,labelRows)-- among  3583. versionn 4.3: 2501  among ~850
  testData, testLabels, testFiles = mergeExperimentsByPolarization(testSamples, testFiles,testData, testLabels)
  print("testData:size()")
  print(testData:size())
  print("testLabels:size()")
  print(testLabels:size())
  print ("new experiments names size is: " .. #testFiles)
  
  if saveDatasetToDisk then
    torch.save('inverseDataset/trainData', trainData)
    torch.save('inverseDataset/testData', testData)
    torch.save('inverseDataset/trainLabels', trainLabels)
    torch.save('inverseDataset/testLabels', testLabels)
  end 
  
  print '==> doing transpose on the data - before sending it to my generic normalization which treats every column as a different feature'
  trainData = trainData:transpose(2,3)
  testData = testData:transpose(2,3)
  numOfFeatures = trainData:size()[3]
  print '==> how many features we have? good question! the answer is: ' 
  print (numOfFeatures)

  if useNormalization then
    print '==> normalize Data!!'
    dofile 'normalizeData.lua'
    print '==> normalization is DONE!!'
  end
end
