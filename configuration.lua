-- global:
seed = 1
threads = 1

cmd = torch.CmdLine()
cmd:text()
cmd:text('SVHN Loss Function')
cmd:text()
cmd:text('Options:')


cmd:text()
opt = cmd:parse(arg or {})

torch.setnumthreads(threads)
torch.manualSeed(seed)
torch.setdefaulttensortype('torch.DoubleTensor')


saveDatasetToDisk = false 
loadDatasetHook = true   
dummyData = false 
fixNoiseAboveZeroPoint9 = true
--DATASET_NAME = 'datasetV6.3'         ----------not augmented and with many many eps 
DATASET_NAME = 'datasetVersion5.6'         ----------not augmented and with many many eps
batchSize = 64  --- better for both directions - use it for verion 5

optimState = {
  learningRate = 0.01,
  momentum = 0.9, --was 0.9
  weightDecay = 0.0005, --was 0.0005
  --  learningRateDecay = 5e-7
  learningRateDecay = 5e-7,
  evalCounter = 0
}

optimStateInverse = {
  learningRate = 0.01,
  momentum = 0.9, --was 0.9
  weightDecay = 0.0005, --was 0.0005
  --  learningRateDecay = 5e-7
  learningRateDecay = 5e-7,
  evalCounter = 0
}

configInverse = {}
--optimMethod = optim.sgd
optimMethod = optim.adadelta

opt.loss ='mse'
--opt.loss ='abs'

--writeDataForStage2 = true
writeDataForStage2 = false
stage2 = false

if stage2 then
  pathToWritePredictions = 'predictions-stage2/'
else
  pathToWritePredictions = 'predictions/'
end


writeWithScore = false
writeResultsToDisk = false
loadNetwork = 0
loadBestTestNetwork = 0
continueLearning = 0  
bothDirectionLoadNetworks = 1 ----if 1, loadNetwork suppose to be 0

backwardDirectAfterXEpochs = 1
backwardInverseEvery = 10000
boost = 0   -- how many epochs to train before testing
testEveryXEpochs = 1
trainPlotEveryXEpochs = 0
plotting = false
polttingAmount = 0
myDebug = false
illustrate = false
saveNormalizationParams = true
normalizeLabels = true
useNormalization = true

filterNoPairedData = false  
whichPolarization = 2   --whichPolarization and filterNoPairedData are relevant for direct function - don't use it for training the inverseFunction should be false
inverseFunction = true
splittedModel = true
bothDirections = False  --if true, turn on also the inverseFunction and splittedModel flag!
modelXY_enabled  = true  --if disabled make sure that you use two different models with input size of 33
GAAlgo = False
useCuda = false

if inverseFunction then
  loadData = '1_data_accurateDataset_inverseFunction.lua'
else
  loadData = '1_data_accurateDataset_directFunction.lua'
end

print '==> loading data'
if stage2 then
  dofile '1_data_accurateDataset_directFunction_stage2.lua'
  dofile '2_model_stage2.lua'


else
  dofile (loadData)

  dofile '2_model_inverse.lua'
  dofile 'buildResidualNetwork.lua'

  dofile 'buildInverseSplittedModel.lua'
  dofile 'buildResidualFullyConnectedNetwork.lua'
--  dofile 'buildSmallModel.lua'
  
  if GAAlgo then
    dofile 'geneticsAlgorithm.lua' 
    exit()
  end
end

noutputs = trainLabels:size()[2]
ninputs = trainData:size()[2] * trainData:size()[3] 
print('trainData:size()')
print(trainData:size())
print('noutputs')
print(noutputs)
print('ninputs')
print(ninputs)
print('trainData[1]')
print(trainData[1])
wait(5) 
if dummyData then

  print '==> dummyData!!'
  trainData = trainData * 0 -- + 0.1
  testData = testData * 0 -- + 0.1
  trainLabels = trainLabels * 0 -- + 0.1
  testLabels = testLabels * 0  -- + 0.1
end

if loadNetwork == 1 then
  print('==>loading network')
  if loadBestTestNetwork == 1 then
    print('==>running tests on best network!!!!!!!!!')
    model = torch.load('bestResultOnTest/model.net')
    
  else
    model = torch.load('results/model.net')
  end
  epoch = 400000

else

  if bothDirections then
    print '===> both directions!'
    if bothDirectionLoadNetworks == 1 then

        modelXY = torch.load('bestResultOnTest/pretrained/model-direct5.6-sameDataAsInverse-0.19MSE.net')
        modelInverse = torch.load('bestResultOnTest/figure4/changing0.93To1/model.net')
        bothNetworks = nn.ParallelTable()--------------------------------fix me to previous line
        bothNetworks:add(modelXY)
        bothNetworks:add(modelInverse) -----------------------------------------------------------TODO fix me
        modelInverse:cuda()
        model = bothNetworks 
    else
      modelInverse = buildInverseSplittedSmallModel(8, 100, 750, 2, 1) 


    bothNetworks = nn.ParallelTable()--------------------------------fix me to previous line
      if modelXY_enabled then
        modelXY = buildSmallModel(-1, -1, 1000,43, 35) --the best for direct?   ----using 40 for training both leades to worst results
        bothNetworks:add(modelXY)----TODO add this line
      else
        modelX = buildSmallModel(-1, -1, 40,43, 33)--TODO it suppose to be 1000 as well?
        modelY = buildSmallModel(-1, -1, 40,43, 33)
        bothNetworks:add(modelX)
        bothNetworks:add(modelY)
      end

     bothNetworks:add(modelInverse)

    model = bothNetworks 
    end 

  else
  
    if inverseFunction then
      if splittedModel then
       model = buildInverseSplittedSmallModel(8, 100, 750, 3, 7) 
      else
         temp = buildLinear(3, 111, 40,0)
         model = nn.Sequential()
         model:add(nn.Reshape(111))
         model:add(temp)
         model:add(nn.Linear(40,8))
        print(model)
      end
    else
     model = buildSmallModelNoDropout(-1, -1, 1000,43, 35) 
    end
  end
end




