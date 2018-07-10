require 'nn'
require 'mattorch'
require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods
require 'nn'      -- provides all sorts of loss functions
require 'image'
require 'mattorch' 
require 'math'
require 'ccn2'
require 'cunn'  --cuda convnet
require 'gnuplot'
dofile 'utils.lua' 
plotting = false
roundGeometryPrediction = true

defaultSpectrumValue = 0.96

mse = nn.MSECriterion()
totalScore = 0


--from random import randint, random
--from operator import add

function initNetworkAndNorms()
  print '==> processing options'
  
  print '==> loading normalization values'
  print '==> loading ground truth'
  
  DLPitch = 700
  labelPitch = 300
  PitchTransform = false
  
  
  print('==>loading network')
   print '==> loading normalization values'
  print('==>loading network')
--  model = torch.load('figure2Model/oldNetwork/queryModel.net')

--  model = torch.load('bestResultOnTest/pretrained/model-5.8-bidirectional-0.26.net')
--  model = torch.load('bestResultOnTest/pretrained/model-5.6-bidirectional-weightSharing.net')


  trainLabelsOriginalStd = torch.load('normalization/direct-trainLabelsStd' ) -- was this 1 - sep 2016
  trainLabelsOriginalMean = torch.load('normalization/direct-trainLabelsMean')
  trainDataStd = torch.load('normalization/direct-trainDataStd')
  trainDataMean = torch.load('normalization/direct-trainDataMean')
--  model = torch.load('bestResultOnTest/queryModel.net')
    model = torch.load('bestResultOnTest/pretrained/model-5.8-bidirectional-0.26.net')
  
 
  print(model)
  model:evaluate()
  model:cuda()
  
  modelXY = model:get(1)
  modelInverse = model:get(2)
  
  --modelInverse = torch.load('bestResultOnTest/pretrained/model-inverse5.6-bigModel-0.06MSE.net')
  modelXY_enabled = true

end


function normelizeTestData(data)
  print("trainDataMean:size()")
  print(trainDataMean)
  
  for i=1,numOfFeatures do
    print(i)
    testData[{ {},{},{i} }]:add(-trainDataMean[i])
    testData[{ {},{},{i} }]:div(trainDataStd[i])
  end
end



useCuda = true
saveDatasetToDiskGA = false
loadDatasetHookGA = true
--DATASET_NAME = 'datasetVersion5.6'         ----------not augmented and with many many eps
--DATASET_NAME = 'datasetVersion5.9'
DATASET_NAME = 'datasetVersion5.6'


require 'math'
math.randomseed( os.time() )

print("run genetics algorithms!")
dofile 'utils.lua'
labelLength = 8

initNetworkAndNorms()
print("trainDataMean:size()")
print(trainDataMean)
trainDataMean = torch.FloatTensor(trainDataMean)
print(trainDataMean:size())
print("trainDataStd:size()")
trainDataStd = torch.FloatTensor(trainDataStd)
print(trainDataStd:size())
 

if (loadDatasetHookGA) then

  --read data from disk
  trainData = torch.load('inverseDataset/trainDataGA5.6')
  testData = torch.load('inverseDataset/testDataGA5.6')
  trainLabels = torch.load('inverseDataset/trainLabelsGA5.6')
  testLabels =  torch.load('inverseDataset/testLabelsGA5.6')
  columns = 1
  rows = 43+43+35-2-8
else

  saveNormalizationParams = true
  normalizeLabels = true
  useNormalization = true
  inverseFunction = true
  splittedModel = true
  bothDirections = true  ---if true, turn on also the inverseFunction and splittedModel flag!
  modelXY_enabled  = true

  loadData = '1_data_accurateDataset_inverseFunction.lua'
  
  dofile (loadData)

  
  numOfFeatures = 111
  print(numOfFeatures)
  print("testData:size()")
  print(testData:size())
  
  dofile 'buildInverseSplittedModel.lua'
end

if saveDatasetToDiskGA then
  torch.save('inverseDataset/trainDataGA5.6', trainData)
  torch.save('inverseDataset/testDataGA5.6', testData)
  torch.save('inverseDataset/trainLabelsGA5.6', trainLabels)
  torch.save('inverseDataset/testLabelsGA5.6', testLabels)
end 

mse = nn.MSECriterion()


function deepcopy(orig)
    local orig_type = type(orig)
    local copy
    if orig_type == 'table' then
        copy = {}
        for orig_key, orig_value in next, orig, nil do
            copy[deepcopy(orig_key)] = deepcopy(orig_value)
        end
        setmetatable(copy, deepcopy(getmetatable(orig)))
    else -- number, string, boolean, etc
        copy = orig
    end
    return copy
end


function individual()
--    'Create a member of the population.'
--    return [ randint(min,max) for x in xrange(length) ]

  l0 = math.random(80,250)
  l1 = math.random(80,250)
  angel = math.random(0,90)
  x1 = math.floor(math.random() + 0.5)
  x2 = math.floor(math.random() + 0.5)
  x3 = math.floor(math.random() + 0.5)
  x4 = math.floor(math.random() + 0.5)
--  x5 = math.floor(math.random() + 0.5)
  x5=1
--  TODO normelize!
  
--  geom =  torch.FloatTensor{x1, x2, x3, x4, l0, x5, l1, angel}
  geom =  {x1, x2, x3, x4, l0, x5, l1, angel}
  print("raw geom encoding")
--  print(torch.round(geom))
--  normedGeom = normelizeGeometry(geom)
--  print(normedGeom)
  return geom
--  return geom
end 

function population(count)
--    """
--    Create a number of individuals (i.e. a population).
--
--    count: the number of individuals in the population
--    length: the number of values per individual
--    min: the minimum possible value in an individual's list of values
--    max: the maximum possible value in an individual's list of values
--
--    """
  pop = {}
  for i=1,count do
    pop[i]  = individual()
    print("pop[i]")
    print(pop[i])
  end
  retVal = torch.FloatTensor(pop) 
  return retVal
end

function fitness(individual, targetX, targetY, inverseData)
--    """
--    Determine the fitness of an individual. Higher is better.
--
--    individual: the individual to evaluate
--    target: the target number individuals are aiming for
--    """
  if(individual[6] == 0) then
    individual[7] = 0
  end
  if( individual[1] == 0 and individual[2] == 0 and individual[3] ==0 and individual[4] == 0) then
    individual[5] = 0
  elseif (individual[5] == 0 ) then
    individual[5] = math.random(80,250)
  end

  normedGeom = normelizeGeometry(individual:clone())
  directX, targetsX  = buildDirectDataNoBatchHook(inverseData, normedGeom, 1)
  directY, targetsY  = buildDirectDataNoBatchHook(inverseData, normedGeom, 2)

  
  predX, normedX =queryDirect(directX,targetX, modelXY, 'x pol', 1)
  predY, normedY =queryDirect(directY,targetY, modelXY, 'y pol', 2)
  scoreX = mse:forward(predX:cuda(), normedX:cuda())
  scoreY = mse:forward(predY:cuda(), normedY:cuda())
  sumScore = scoreX + scoreY
--  print("sum scode is: " )
--  print(sumScore)
  return sumScore
--    sum = reduce(add, individual, 0)
--    return abs(target-sum)
end 

function grade(pop, targetX, targetY, inverseData)
--  'Find average fitness for a population.'
  summed = 0
  for i=1, #pop do
    summed = summed + fitness(pop[i], targetX, targetY, inverseData) 
  end
  return summed / (#pop * 1.0)
end


retain=0.2
random_select=0.05
mutate=0.05

function evolve(pop, targetX, targetY, inverseData) 

  
  graded= {}
  
  score = 0
  for i=1,pop:size()[1] do
    fitVal = fitness(pop[i],targetX, targetY, inverseData)

    temp = {fitVal,pop[i]}
    
    graded[i] = temp
    score = score + fitVal
  end
  print("average score for generation:")
  print(score/pop:size()[1])
  table.sort( graded, cmp_multitype )--make sure it is sorted in the right direction
  retain_length = math.floor(#graded * retain)
  parents = {}
  parGrades = {}
  for i=retain_length,pop:size()[1] do
    table.insert(parents, graded[i][2])
    table.insert(parGrades, graded[i][1])
  end
  
  for individual = 1, retain_length do
      tempRand = math.random()
      if random_select > tempRand then
          table.insert(parents, graded[individual][2])
          table.insert(parGrades, graded[individual][1])
      end
  end


  for i=1,#parents do
    if mutate > math.random() then
      l0r = (math.random()/2+0.75)--/0.5*2
      l1r = (math.random()/2+0.75)--/0.5*2
      angelr = (math.random()/2)+0.75--+0.5
      edge = math.random(1,4)
      parents[i][edge] = (parents[i][edge] + 1) % 2
      parents[i][5] = parents[i][5] * l0r-- math.random(80,250)
      parents[i][7] = parents[i][7] * l1r --math.random(80,250)
      parents[i][8] =parents[i][8] * angelr  --math.random(0,90)
      if (parents[i][5] > 300) then
        parents[i][5] = 300
      end
      if(parents[i][7] > 300) then
        parents[i][7] = 300
      end
      if(parents[i][8] > 90) then
        parents[i][8] = 90
      end
      mutateFit = fitness(parents[i],targetX, targetY, inverseData)

      parGrades[i] = mutateFit

    end
  end

  parents_length = #parents
  desired_length = pop:size()[1]
  poorGenetics = 0
  while #parents < desired_length do
--    print("in crossoverrrrrrrrrrrrrrr")
    maleInd = math.floor(math.random(1, parents_length))
    femaleInd = math.floor(math.random(1, parents_length))
    male = parents[maleInd]
    female = parents[femaleInd]
    if(not torch.all(torch.eq(male, female))) then 
      child = deepcopy(male)
      for i=1,8 do --randomly mix both male and female properties
        rand = math.floor(math.random() + 0.5)
        if (rand == 1) then 
          child[i] = female[i]
        end
      end
      table.insert(parents, child)
      childFit =  fitness(child,targetX, targetY, inverseData)
      table.insert(parGrades,childFit)
      
    else

      poorGenetics = poorGenetics + 1
      print(mutate)

      if (poorGenetics% 20 == 0 and mutate < 0.17) then
        mutate = mutate + 0.02
        print("mutate ratio is: ")
        print(mutate)
        wait(5)
      end
      print("miss - generation geneticsis poor")
    end
  end
  return parents, parGrades

end



function cmp_multitype(op1, op2)
  return op1[1] > op2[1] 
end

function __genOrderedIndex( t )
    local orderedIndex = {}
    for key in pairs(t) do
        table.insert( orderedIndex, key )
    end
    table.sort( orderedIndex, cmp_multitype ) --### CANGE ###
    return orderedIndex
end

function normelizeGeometry(input)
  for i=1,8 do
    input[i] = input[i] -trainLabelsOriginalMean[i]
    input[i] = input[i] / trainLabelsOriginalStd[i]
  end
  return input
end

function queryBothDirectionsNetwork(direct)


  print '==> normelize test data'
  fixFabrication()
  
  testData = torch.FloatTensor(1,43+43+25)
  temp = torch.cat(direct[{{1+8,25+8}}]:float(), transY, 1)
  testData[1] = torch.cat(temp, transX:float(), 1)
  numOfFeatures = 111
  labelLength = 8
  normelizeTestData(testData)

  print '==> build inverse data'
  inverseData = torch.FloatTensor(3,43)
  
  inverseData[1] =  torch.cat(testData[1][{{1,25}}]:float(), torch.zeros(18):float(),1)
  inverseData[2] = testData[1][{{26,26+42}}]
  inverseData[3] = testData[1][{{26+43,26+43+42}}]
  
  geometryRaw = queryInverse(inverseData,groundTruth, modelInverse)
  geometry = inverseNormalizationOnLabel(geometryRaw:clone())
  roundGeometry = geometryRaw:clone()
  roundGeometry = inverseNormalizationOnLabel(roundGeometry)
  roundGeometry = torch.round(roundGeometry)
  print (roundGeometry)
  roundGeometry = normelizeTestLabels(roundGeometry)
  print ("round inverse DNN prediction")
  print (geometry)
  local directX, targetsX  = buildDirectDataNoBatch(inverseData, roundGeometry, 1)
  local directY, targetsY  = buildDirectDataNoBatch(inverseData, roundGeometry, 2)
  print '==> query bidirectional network'
  predX = queryDirect(directX,transX, modelXY, 'bidirectional-x ' .. direct[33],1)
  predY = queryDirect(directY,transY, modelXY, 'bidirectional-y ' .. direct[33],2)
  
  
  return geometry, predX, predY, mse:forward(predX:float(),transX:float()) + mse:forward(predY:float(),transY:float()) 

end

--build waveLengths
waveLengths = torch.FloatTensor(43)
for i=1,43 do
  waveLengths[i] = 575 +i*25
end
print(waveLengths)


function writeBatchResultTo(path, prediction, filenames, index, groundTruth)
  print("do nothing")
end

function inverseNormalizationOnLabel(pred)
  for i=1,labelLength do
    --        print '(pred[1][i])'
    --        print (pred[1][i])
    pred[i] = pred[i] * trainLabelsOriginalStd[i]
    pred[i] = pred[i] + trainLabelsOriginalMean[i]

  end
  return pred
end


function normalizateTestLabel(pred)
  for i=1,labelLength do
    --        print '(pred[1][i])'
    --        print (pred[1][i])
    pred[i] = pred[i] - trainLabelsOriginalMean[i]
    pred[i] = pred[i] / trainLabelsOriginalStd[i]

  end
  return pred
end


function queryDirect(input,label, modelArg, title, pol)
  rawOutput = modelArg:forward(input:clone():cuda())
  local output = inverseNormalizationTranmission(rawOutput,pol)
  local labelNormed = inverseNormalizationTranmission(label:clone(),pol)
  
  plottingPrediction = true
  plot(labelNormed,output,output*0,title, true) --TODO use this line to plot!!!
  return output:clone(), labelNormed:clone()
end


function queryInverse(input,label, modelArg)
  rawOutput = modelArg:forward(input:cuda())

  if PitchTransform then
    output = torch.pow(output, (DLPitch)/(labelPitch))
  end
  return rawOutput
end



DLPred = {}
GAPred = {}--torch.DoubleTensor(8):reshape(1,8)--{}
GT = {}--torch.DoubleTensor(8):reshape(1,8)--{}
mseDL = {}
mseGA = {}

--iterate over the test set - for each sample use GA to predict a good geometry
for testSampleIndex=1,testData:size()[1] do
  input = testData[testSampleIndex]
  print("test data is: ")
  
  target = testLabels[testSampleIndex]
  targetX = input[{3,{}}]
  targetY = input[{2,{}}]
  inverseData = buildInverseDataNoBatch(targetX,targetY, input)
  
  popSize = 100
  generationSize = 100
  pop = population(popSize)
  candidate = torch.FloatTensor()
  for generationIndex =1,generationSize do
    print("generation: ")
    print(generationIndex)
    print("test sample: ")
    print(testSampleIndex)
    
    pop, popGrades = evolve(pop, targetX, targetY, inverseData)
    grades = torch.FloatTensor(popGrades)
    popTensor = pop[1]:reshape(1,8)
    popRaw = pop[1]:reshape(1,8)

    for j=2,popSize do
      popRaw = torch.cat(popRaw, torch.FloatTensor(pop[j]):clone():reshape(1,8), 1)
      popTensor = torch.cat(popTensor, torch.FloatTensor(pop[j]):reshape(1,8), 1)
    end
    pop = popTensor
    everyIter = 1
    if (generationIndex % everyIter == 0) then
      if (generationIndex == everyIter) then
        min = grades[1]
        minGeom = popTensor[{1}]
        minGeomHack = popTensor[{1}]
        minGeomNormedHack = normelizeGeometry(minGeomHack:clone())  
        normedTarget = normelizeGeometry(target:clone())
        minHack = mse:forward(minGeomNormedHack, target)
        minHackSpecGrade = grades[1]
      end
      local MinGeomIndex = -1
      for k=1,grades:size()[1] do
       
        if (grades[{k}] < min) then
          MinGeomIndex, min = k, grades[{k}]
          minGeom = popTensor[{MinGeomIndex}]
        end

        hackValue  =  mse:forward(normelizeGeometry(popTensor[{k}]:clone()), normedTarget)
        if(hackValue  < minHack) then
          minGeomHack = popTensor[{k}]
          minHack = hackValue 
          minHackSpecGrade = grades[k]
     
        end
        
      end

     
      if (generationIndex % 1 == 0) then

        print("total score is: ")
        print (totalScore / testSampleIndex)

        
      end
    end

  end

  plotting = false

  print('testData[testSampleIndex]')
  print(testData[testSampleIndex])
  
  print('inverseData')
  print(inverseData)
  
  geometryRaw = queryInverse(inverseData,groundTruth, modelInverse)
  geometry = inverseNormalizationOnLabel(geometryRaw:clone())
  roundGeometry = geometryRaw:clone()
  roundGeometry = inverseNormalizationOnLabel(roundGeometry)
  roundGeometry = torch.round(roundGeometry)
  print ("before rounding")
  print (geometry)
  print ("round inverse DNN prediction")
  print (roundGeometry)
  dlGeom = roundGeometry:clone()
  roundGeometry = normelizeTestLabels(roundGeometry:clone())


  directX, targetsX  = buildDirectDataNoBatch(inverseData, roundGeometry, 1)
  directY, targetsY  = buildDirectDataNoBatch(inverseData, roundGeometry, 2)
  predX = queryDirect(directX,targetsX, modelXY, 'bidirectional-x ' ,1)
  predY = queryDirect(directY,targetsY, modelXY, 'bidirectional-y ',2)
  torch.save( 'results/retreived-xDNN' , predX:double(),'ascii')
  torch.save( 'results/retreived-yDNN', predY:double(),'ascii')
--end of query inverse!!!!!!!
  
  normedGeom = normelizeGeometry(minGeom:clone())
  directX, targetsX  = buildDirectDataNoBatchHook(inverseData, normedGeom, 1)
  directY, targetsY  = buildDirectDataNoBatchHook(inverseData, normedGeom, 2)


  predX, normedX =queryDirect(directX,targetsX, modelXY, 'x pol', 1)
  predY, normedY =queryDirect(directY,targetsY, modelXY, 'y pol', 2)


     
  torch.save( 'results/desired-x' , normedX:double(),'ascii')
  torch.save( 'results/desired-y' , normedY:double(),'ascii')
  torch.save( 'results/retreived-xGA' , predX:double(),'ascii')
  torch.save( 'results/retreived-yGA', predY:double(),'ascii')



  normedGeomHack = normelizeGeometry(minGeomHack:clone())
  directXHack, targetsXHack  = buildDirectDataNoBatchHook(inverseData, normedGeomHack, 1)
  directYHack, targetsYHack  = buildDirectDataNoBatchHook(inverseData, normedGeomHack, 2)
  predXHack, normedXHack =queryDirect(directXHack,targetsXHack, modelXY, 'x pol Hack', 1)
  predYHack, normedYHack =queryDirect(directYHack,targetsYHack, modelXY, 'y pol Hack', 2)
  scoreXHack = mse:forward(predXHack:cuda(), normedX:cuda())
  scoreYHack = mse:forward(predYHack:cuda(), normedY:cuda())
  sumScoreHack = scoreXHack + scoreYHack
  print("mse on hack geom spectrum")
  print(sumScoreHack)

  scoreX = mse:forward(predX:cuda(), normedX:cuda())
  scoreY = mse:forward(predY:cuda(), normedY:cuda())
  sumScore = scoreX + scoreY
  print("min geom")
  print(torch.round(minGeom:clone()))
  print("ground truth is: ")
  print(inverseNormalizationOnLabel(target))
  normedTarget = normelizeGeometry(target:clone())

  print("normed min geom:")
  print(normedGeom)
  print("normed target geom: " )
  print(normedTarget)


  print("mse on spectrum: ")
  print(sumScore)
  print("GA mse on geom: ")
 
  score = mse:forward(normedTarget, normedGeom)
  print(score)
  
  print("DL mse on geom")
  dlMSEGeom = mse:forward(normedTarget,roundGeometry:float())
  print(dlMSEGeom)
  
  totalScore = totalScore + score
  print("total score is: ")
  print (totalScore / testSampleIndex)
  plotting = false
--  wait(20)


  --save the prediction in order to calculate paired t test and p value
  print(type(roundGeometry))
  print((roundGeometry))
  
  print(roundGeometry:double())
  print(type(roundGeometry:double()))
  table.insert(DLPred, torch.totable(dlGeom:double()))
  table.insert(GAPred, torch.totable(torch.round(minGeom:clone()):double()))
  table.insert(GT, torch.totable(torch.round(target):double()))
  print(type(dlMSEGeom))
  table.insert(mseDL,{dlMSEGeom})
  table.insert(mseGA, {score})

end

print(DLPred)
csvigo.save('results/GA-DLPred',  DLPred)
csvigo.save('results/GA-GAPred',  GAPred)
csvigo.save('results/GA-GT',  GT)
csvigo.save('results/GA-mseDL', mseDL)
csvigo.save('results/GA-mseGA',  mseGA)
