
function splitData(data, newData,labels,newLabels, columns,rows, padding)


  print (data[{1,1,{}}])

  for i=1,data:size()[1] do
    newData[i][1] =  torch.cat(data[{i,1,{1,25}}], torch.zeros(18):float())
    newData[i][2] = data[{i,1,{26,68}}]
    newData[i][3] = data[{i,1,{69,111}}]
  end
end

if splittedModel then
  print "==>chaning the data to split model..."
  wait(5)

  columns = 3
  rows = 43
  labelRows = 8
  padding = 1 --not used for now

  tempTrainData = torch.FloatTensor(trainData:size()[1],columns,rows)--4819
  tempTestData = torch.FloatTensor(testData:size()[1], columns,rows)--1791

  splitData(trainData, tempTrainData,testLabels,tempTestLabels, columns,rows, padding)
  splitData(testData, tempTestData,trainLabels,tempTrainLabels, columns,rows, padding)
  trainData = tempTrainData
  testData = tempTestData
  print("testData[1]")
  print(testData[1])
  wait(10)
  print("trainData[1]")
  print(trainData[1])

  print("testData[1]")
  print(testData[1])

end


function buildInverseSplittedModel(noutputs, n1, n2, n3, joinLayer)

  print('building model....')
  factor = 4 

  local model = nn.Sequential()       -- Create a network that takes a Tensor as input
  
  model:add(nn.SplitTable(1,2))  -- split by first column - each experiment has 3 groups of data - first column has padding with zeros
  c = nn.ParallelTable()      -- The two Tensor slices go through two different Linear
  c:add(buildLinear(3,43,250,0))     -- Layers in Parallel
  c:add(buildLinear(3,43,250,0))
  c:add(buildLinear(3,43,250,0))
  
  model:add(c)                  -- Outputing a table with 2 elements
  model:add(nn.JoinTable(1,1))    -- Finally, the tables are joined together and output.

  model:add(nn.Linear(750,750))
  model:add(nn.ReLU())
  model:add(nn.Linear(750,750))
  model:add(nn.ReLU())
  
  model:add(nn.Linear(750,750))
  model:add(nn.ReLU())
  model:add(nn.Linear(750,750))
  model:add(nn.ReLU())
  model:add(nn.Linear(750,750))
  model:add(nn.ReLU())
  model:add(nn.Linear(750,750))
  model:add(nn.ReLU())

  model:add(nn.Linear(750,noutputs))

  print('model is ready')
  print('==> printing model...')
  print(model)

  return model
end

function buildInverseSplittedSmallModel(noutputs, n1, n2, d1, joinLayer)

  print('building model....')
  factor = 4 

  local model = nn.Sequential()  --Create a network that takes a Tensor as input
  
  --else, if you want to use parallel architecture
  model:add(nn.SplitTable(1,2))  -- split by first column - each experiment has 3 groups of data - first column has padding with zeros
  c = nn.ParallelTable()         -- The two Tensor slices go through two different Linear
  c:add(buildLinear(d1,43,n1,0))  -- Layers in Parallel
  c:add(buildLinear(d1,43,n1,0))
  c:add(buildLinear(d1,43,n1,0))
  model:add(c)                  -- Outputing a table with 2 elements
  model:add(nn.JoinTable(1,1))    -- Finally, the tables are joined together and output.
  
  model:add(nn.Linear(n1*3,n2))
  model:add(nn.ReLU())
  for i = 2 ,joinLayer do
    model:add(nn.Linear(n2,n2))
    model:add(nn.ReLU())
  end
  
  model:add(nn.Linear(n2,noutputs))

  print('model is ready')
  print('==> printing model...')
  print(model)
  wait(5)

  return model
end

function  buildLinear(nLayers,firstLayerSize, layerSize,dropoutAmount)
  local linear = nn.Sequential()

  linear:add(nn.Linear(firstLayerSize,layerSize))
  linear:add(nn.ReLU())
  linear:add(nn.Dropout(dropoutAmount)) -- used it was 0.5

  for i=2,nLayers do
    linear:add(nn.Linear(layerSize,layerSize))
    linear:add(nn.ReLU())
    linear:add(nn.Dropout(dropoutAmount)) -- used it was 0.5
  end

  return linear

end

function  buildLinear(nLayers,firstLayerSize, layerSize,dropoutAmount)
  local linear = nn.Sequential()

  linear:add(nn.Linear(firstLayerSize,layerSize))
  linear:add(nn.ReLU())
  linear:add(nn.Dropout(dropoutAmount)) -- used it was 0.5

  for i=2,nLayers do
    linear:add(nn.Linear(layerSize,layerSize))
    linear:add(nn.ReLU())
    linear:add(nn.Dropout(dropoutAmount)) -- used it was 0.5
  end

  return linear

end

function buildConv(nLayers,firstLayerSize, layerSize, kw,dropoutAmount)
  local conv = nn.Sequential()

  conv:add(nn.TemporalConvolution(43,40,1,1)) --43 goes in, 36X10 goes out
  conv:add(nn.TemporalMaxPooling(2))       --36X10 goes in, 18X10 goes out
  conv:add(nn.ReLU())
  conv:add(nn.Dropout(dropoutAmount)) -- used it was 0.5
  
  

  return conv
end
