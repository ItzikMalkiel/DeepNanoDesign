-------------------------util---------------------------------------

require 'csvigo'
require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods
require 'nn'      -- provides all sorts of loss functions
require 'image'
--require 'mattorch' 
require 'math'

--require 'nnx'      --
--if opt.type == 'cuda' then -- always use cude
if useCuda then
  require 'ccn2'
  require 'cunn'  --cuda convnet
end
  --require 'nn2'
--end


absCriterion = nn.AbsCriterion()
absCriterion.sizeAverage = false


function wait(seconds)
  local start = os.time()
  repeat until os.time() > start + seconds
end

function printDebug(string)
  if myDebug then
    print(string)
  end
end

function waitDebug(seconds)
  if myDebug then
    wait(seconds)
  end
end

fixNoiseAboveZeroPoint9 = true

function plot(target, pred, diff, string, render)

  string = string or 'diff'
  render = render or false    
  
  if plotting then
          require 'gnuplot'
         localTarget = target
         localPred = pred
         localDiff = diff:double()
--          print("pred")
--          print(pred)
          if localPred:size()[1] ~= 43 then
            localPred = localPred:double()[{1,{1,43}}]
          end
          if localTarget:size()[1] ~= 43 then
            localTarget = localTarget:double()[{1,{1,43}}]
          end
           if localDiff:size()[1] ~= 43 then
            localDiff = localDiff:double()[{1,{1,43}}]
          end
--          print('toPlot') 
--          print(toPlot)
          if stage2 or waveLengths:size():size() == 1 then
            XValues = waveLengths
          else
            XValues = waveLengths[{{1,43},1}]
--            print('target') 
--            print(target)  
          end
--          print('XValues') 
--          print(XValues) 
          
          --          gnuplot.figure()
          --          gnuplot.epsfigure('figures/'..string)  
          if render then
            gnuplot.figure()
          else       
              if writeWithScore then
                --calculate the inverse value of the ground truth - but DONT accedently chagne the real ground truth value!
                --write the prediction with an abs score (comparing to groundtruth)
                gnuplot.pdffigure('figures/'.. absCriterion:forward(localPred:clone():cuda(),localTarget:clone():cuda()) ..string)
              else      
--                print (string)                     
                 gnuplot.pdffigure('figures/'..string)
              end
          end
          gnuplot.raw('set yrange [0:1.3]')
          gnuplot.xlabel('wavelength')
          gnuplot.ylabel('transmission')
          gnuplot.plot({'target', XValues, localTarget, '-'}, {'prediction',XValues,localPred,'+'} , {string, XValues, localDiff, '+'})
          print("diff abs value")
          temp = absCriterion(localDiff*0, localDiff)
          print(temp)
          
--          gnuplot.epsfigure(string)
--          gnuplot.plot({'Sin Curve',torch.sin(torch.linspace(-5,5))})
          gnuplot.plotflush()
--          ({'Cos',x/math.pi,torch.cos(x),'~'},{'Sin',x/math.pi,torch.sin(x),'|'})
--          wait(0.5)

          --write raw data to file
          local list = {waveLengths = XValues:double(), comsol = localTarget:double(), deepLearning = localPred:double()}
          mattorch.save( 'figures/rawResult_'  .. string, list)
      end

end



function plotFigure3(left, test, right, string, render)

  string = string or 'diff'
  render = render or false    
  
  if plotting then
          require 'gnuplot'
         localTarget = left
         localPred = test
         localDiff = right:double()
--          print("pred")
--          print(pred)
          if localPred:size()[1] ~= 43 then
            localPred = localPred:double()[{1,{1,43}}]
          end
          if localTarget:size()[1] ~= 43 then
            localTarget = localTarget:double()[{1,{1,43}}]
          end
           if localDiff:size()[1] ~= 43 then
            localDiff = localDiff:double()[{1,{1,43}}]
          end
--          print('toPlot') 
--          print(toPlot)
          if stage2 or waveLengths:size():size() == 1 then
            XValues = waveLengths
          else
            XValues = waveLengths[{{1,43},1}]
--            print('target') 
--            print(target)  
          end
--          print('XValues') 
--          print(XValues) 
          
          --          gnuplot.figure()
          --          gnuplot.epsfigure('figures/'..string)  
          if render then
            gnuplot.figure()
          else       
              if writeWithScore then
                --calculate the inverse value of the ground truth - but DONT accedently chagne the real ground truth value!
                --write the prediction with an abs score (comparing to groundtruth)
                gnuplot.pdffigure('figures/'.. absCriterion:forward(localPred:clone():cuda(),localTarget:clone():cuda()) ..string)
              else      
--                print (string)                     
                 gnuplot.pdffigure('figures/'..string)
              end
          end
          gnuplot.raw('set yrange [0:1.3]')
          gnuplot.xlabel('wavelength')
          gnuplot.ylabel('transmission')
          gnuplot.plot({'left', XValues, localTarget, '+'}, {'test',XValues,localPred,'+'} , {"right", XValues, localDiff, '+'})
          print("diff abs value")
          temp = absCriterion(localDiff*0, localDiff)
          print(temp)
          
--          gnuplot.epsfigure(string)
--          gnuplot.plot({'Sin Curve',torch.sin(torch.linspace(-5,5))})
          gnuplot.plotflush()
--          ({'Cos',x/math.pi,torch.cos(x),'~'},{'Sin',x/math.pi,torch.sin(x),'|'})
--          wait(0.5)

          --write raw data to file
          local list = {waveLengths = XValues:double(), comsol = localTarget:double(), deepLearning = localPred:double()}
          mattorch.save( 'figures/rawResult_'  .. string, list)
      end

end

function plotOnePolarization(target, string, render, prediction, plottingPrediction)

  string = string or 'diff'
  render = render or false    
  
  if plotting then
          require 'gnuplot'
         localTarget = target
         localPredciction = prediction
--          print("pred")
--          print(pred)
          if localPredciction:size()[1] ~= 43 then
            localPredciction = localPredciction:double()[{1,{1,43}}]
          end
          if localTarget:size()[1] ~= 43 then
            localTarget = localTarget:double()[{1,{1,43}}]
          end
--          print('toPlot') 
--          print(toPlot)
          if stage2 or waveLengths:size():size() == 1 then
            XValues = waveLengths
          else
            XValues = waveLengths[{{1,43},1}]
--            print('target') 
--            print(target)  
          end
--          print('XValues') 
--          print(XValues) 
          
          --          gnuplot.figure()
          --          gnuplot.epsfigure('figures/'..string)  
          if render then
            gnuplot.figure()
          else       
              if writeWithScore then
                --calculate the inverse value of the ground truth - but DONT accedently chagne the real ground truth value!
                --write the prediction with an abs score (comparing to groundtruth)
                gnuplot.pdffigure('figures/'.. absCriterion:forward(localPred:clone():cuda(),localTarget:clone():cuda()) ..string)
              else                           
                 gnuplot.pdffigure('figures/'..string)
              end
          end
          
          gnuplot.raw('set yrange [0:1]')
          gnuplot.raw('set xrange [600:1650]')
          
--          gnuplot.axis{0,1,2,3}
--          gnuplot.xlabel('wavelength')
--          gnuplot.ylabel('transmission')
          gnuplot.grid(true)
          if plottingPrediction then
            gnuplot.plot({ XValues, localTarget}, {XValues, prediction})
          else
            gnuplot.plot({ XValues, localTarget})
          end
          print("diff abs value")
          
--          gnuplot.epsfigure(string)
--          gnuplot.plot({'Sin Curve',torch.sin(torch.linspace(-5,5))})
          gnuplot.plotflush()
--          ({'Cos',x/math.pi,torch.cos(x),'~'},{'Sin',x/math.pi,torch.sin(x),'|'})
--          wait(0.5)

          --write raw data to file
--          local list = {waveLengths = XValues:double(), comsol = localTarget:double(), deepLearning = localPred:double()}
--          mattorch.save( 'figures/rawResult_'  .. string, list)
      end

end

function initData(data, labels, samples)
  print (samples[1].output_data)

  for i = 1,#samples do
    
    --TODO fix this code if needed
    y = samples[i].output_data or samples[i]
    experimentVariables = y[{{1,35}}]
--    experimentVariables = y[{{1,42}}]
    
    waveLengths =  y[{{36,78}}]
    transmittions = y[{{79,121}}] 
    R = y[{{122,164}}]

--    experimentVariables[33] = 1.0

    data[{{i},{}}] = experimentVariables
    labels[{{i},{}}] = transmittions
  end
end

function saveNetwork(path)
  local filename = paths.concat(path, 'model.net')
  os.execute('mkdir -p ' .. sys.dirname(filename))
  print('==> saving model to '..filename)
  torch.save(filename, model)
end


function writeBatchResultTo(path, prediction, filenames, index, groundTruth)

  if writeResultsToDisk then
    printDebug('debug: writing prediction...')
    printDebug(prediction)
    printDebug('debug: index: ' .. index)
    
  
    onePred = prediction:double()
  
    filename = filenames[index].. '.mat'
    
    if writeWithScore then
      --calculate the inverse value of the ground truth - but DONT accedently chagne the real ground truth value!
      tempGroundTruth = torch.DoubleTensor(groundTruth:size()):copy(groundTruth:double())
      inverseNormalizationOnLabel(tempGroundTruth)
      --write the prediction with an abs score (comparing to groundtruth)
      mattorch.save( path  .. absCriterion:forward(onePred,tempGroundTruth) .. "_" .. filename, onePred)
    else
      mattorch.save( path  .. filename, onePred)
    end
    
    printDebug(filename)
    printDebug(torch.round(onePred))
  --  printDebug(absCriterion:forward(onePred,tempGroundTruth))
    printDebug('**************************************************************')
  end
end




fixVector = {0.835811550798274,0.844624207821113,0.852235844386737,0.862246703212434,0.852217191175454,0.863878928210388,0.868176865819822,0.872698841342118,0.876621454934900,0.880243191614872,0.883825906269280,0.887040221754339,0.890225835628220,0.893230716838799,0.895992392732880,0.898356857740738,0.900731171482952,0.903222522960663,0.905190914551090,0.907057932307392,0.909205357214086,0.911331854194739,0.913263393500493,0.914449551549316,0.915562000756548,0.917086073073366,0.918549824496742,0.919910328826845,0.921034965253948,0.922088372150407,0.923292525362469,0.924812694550916,0.926002791074346,0.926948437198502,0.927868798213743,0.928777672263030,0.929653005681035,0.930489728219776,0.931296524225784,0.932079458483331,0.932841836382265,0.933587724891203,0.934322884248625}
fixVector = torch.DoubleTensor(fixVector)
waveLengths = torch.FloatTensor(43)
for i=1,43 do
  waveLengths[i] = 575 +i*25
end
print(waveLengths)

print('==> defining loadData function')

function loadDataFromPath(dataset, rawData, Files)

  ----------------------------------------------------------------------
  -- 2. Load all filetrainFiless in directory
  ext = 'mat'
  
  tempDataFiles = {}

  -- Go over all files in directory. We use an iterator, paths.files().
  for file in paths.files(dataset) do
    -- We only load files that match the extension
    if file:find(ext .. '$') then
      -- and insert the ones we care about in our table
      table.insert(tempDataFiles, file)
    end
  end

  print('==> amount of file found is:')
  print(#tempDataFiles)


  -- Check files
  if #tempDataFiles == 0 then
    error('given directory doesnt contain any files of type: ' .. ext)
  end

  ----------------------------------------------------------------------
  -- We sort files alphabetically, it's quite simple with table.sort()
  -- no need to sort for now
  -- --TODO it doesn't sort the files - fix it in the future if needed
  --table.sort(files, function (a,b) return a < b end)

  print('==> Found files. for example, this is the first file:')
  print(tempDataFiles[1])

  ----------------------------------------------------------------------

  -- Go over the file list:
  for i,file in ipairs(tempDataFiles) do
    filename = tempDataFiles[i]
    filePath = dataset .. filename 
    sample =  mattorch.load(filePath)

    if sample.output_data[1][1] == 0 and sample.output_data[2][1] == 0 and sample.output_data[3][1] == 0 and sample.output_data[4][1] == 0 then 
      sample.output_data[5] = 0
    end
    
    if sample.output_data[6][1] == 0 then
      sample.output_data[7] = 0
    end

    if sample.output_data[33][1] == 1.0 then
--    
      sample.output_data[{{79,121},{1}}] = torch.cmul(fixVector,sample.output_data[{{79,121},{1}}]) ----------------------------------------
    end
     
    table.insert(rawData, sample)
    table.insert(Files, filename)
  end

  print('printing a sample...')
  print(sample)
end


----------------------------------------------------------------------


function buildDirectData(inputs, targets, polarization)
  local size = 33
  if  modelXY_enabled then
    size = 35
  end
  
  local direct = torch.FloatTensor(inputs:size()[1], size)
  local transmission = torch.FloatTensor(inputs:size()[1], 43)

  for i = 1,inputs:size()[1] do
    direct[{i,{1,33}}]  =  torch.cat(targets[{i,{1,8}}]:float(), inputs[{i,1,{1,25}}]:float(),1) --geometries + material and eps and pitch
    if (polarization == 2) then --polarizaion y is the second and x is the third
      if  modelXY_enabled then
        direct[i][34] = -1
        direct[i][35] = 1
      end
      transmission[i][{}] = inputs[{i,2,{}}]
    else
      if  modelXY_enabled then
        direct[i][34] = 1-------------------------------------------------fix me
        direct[i][35] = -1
      end
      transmission[i][{}] = inputs[{i,3,{}}]
    end
--    direct[i][1] = torch.cat(temp,pol,1)
  end

  return direct, transmission
end


function buildDirectDataNoBatch(inputs, targets, polarization)
  print("in buildDirectDataNoBatch")
  local size = 33
  if  modelXY_enabled then
    size = 35
  end
  
  local direct = torch.FloatTensor(size)
  local transmission = torch.FloatTensor(43)

    direct[{{1,33}}]  =  torch.cat(targets[{{1,8}}]:clone():float(), inputs[{1,{1,25}}]:float(),1) --geometries + material and eps and pitch
    if (polarization == 2) then --polarizaion y is the second and x is the third
      if  modelXY_enabled then
        direct[34] = -1
        direct[35] = 1
      end
      transmission[{}] = inputs[{2,{}}]
    else
      if modelXY_enabled then
        direct[34] = 1-------------------------------------------------fix me
        direct[35] = -1
      end
      transmission[{}] = inputs[{3,{}}]
    end
--    direct[i][1] = torch.cat(temp,pol,1)

  return direct, transmission
end

function buildDirectDataNoBatchHook(inputs, targets, polarization)
  local direct = torch.FloatTensor(35)
  local transmission = torch.FloatTensor(43)

    direct[{{1,33}}]  =  torch.cat(targets[{{1,8}}]:float(), inputs[{1,{1,25}}]:float(),1) --geometries + material and eps and pitch
    if (polarization == 2) then --polarizaion y is the second and x is the third
--      pol = {1,0}
      direct[34] = -1
      direct[35] = 1
      transmission[{}] = inputs[{2,{}}]
    else
      direct[34] = 1-------------------------------------------------fix me
      direct[35] = -1
      transmission[{}] = inputs[{3,{}}]
    end
--    direct[i][1] = torch.cat(temp,pol,1)

  return direct, transmission
end


function buildInverseDataNoBatch(outputX, outputY, inputs)
  local direct = torch.FloatTensor(columns,rows)
  local transmission = torch.FloatTensor( 8)
  local newData = torch.FloatTensor(3,43)
  

  for i=1,inputs:size()[1] do
    newData[{1,{}}] =  torch.cat(inputs[{1,{1,25}}], torch.zeros(18):float(),1)
--    newData[{i,1,{}}] =  inputs[{i,1,{}}]
    newData[{2,{}}] = outputY[{}]:float()
    newData[{3,{}}] = outputX[{}]:float()
  end
  
  return newData
end



function fixNoiseAboveZeroPoint9Func(trans)
  for i= 1,43 do
    if trans[i][1] > 0.93 then
      trans[i][1] = 1
    end
  end
  
  return trans
end

--cut filename by "polarization_"
--change polarization to the other one 
-- make sure you always use the same pattern - polarization X and then y - and remove these from data
function mergeExperimentsByPolarization(samples, rawFiles,data, labels)
  print("samples[1].output_data")
  print (samples[1].output_data)
  print("rawFiles[1]")
  print (rawFiles[1])

  print("samples[2].output_data")
  print (samples[2].output_data)
  print("rawFiles[2]")
  print (rawFiles[2])


  print "creating new array to hold the merged-new experiments names"
  newFileNames = {}


  pattern = "_polarization_"
  patternLen = pattern:len()

  foundCounter = 0
  if illustrate then
    howManyToLoad = batchSize
  else
    howManyToLoad = #rawFiles
  end

  for j= 1,howManyToLoad do
    found = false
    --get x polarization experiment representation - we also check for equality in the experiment representation due to data augmentation 
    --that can lead to two matching experiments with different prefix   
    firstExperiment = samples[j].output_data
    matching = firstExperiment[{{1,35}}]
    matching = torch.Tensor(matching:size()):copy(matching)

    if torch.all(firstExperiment[34]:eq(1))then


      experiment = rawFiles[j]
      --print "experiment"
      --print (experiment) 

      startIndex = string.find(experiment,pattern)
      prefix = string.sub(experiment,0,startIndex+patternLen-1)
      --print("prefix is")
      --print(prefix)

      secondExperimentName = prefix .. "x0y1_P_700.mat"
      --print("secondExperimentName")
      --print(secondExperimentName)

      --look for the second experiment
      --remember not to use the second experiment during the iterations
      found = false

      matching[34] = 0
      matching[35] = 1.0
      --exit()
      for i = 1,#rawFiles do
        
        if found then  ---------new code to avoid duplication of the same merged experiments when there are two or more matches in the dataset
          break
        end 
  
        -- torch.all((x*0):eq(y*0)) --compare two tensors 
        secondExperiment = samples[i].output_data
        experimentVariables = secondExperiment[{{1,35}}]

        --        if (rawFiles[i] == secondExperimentName) or (torch.all(matching:eq(experimentVariables)))  then
        if torch.all(matching:eq(experimentVariables)) then
          found = true
          foundCounter = foundCounter + 1


          waveLengths =  secondExperiment[{{36,78}}]
          transmittionsY = secondExperiment[{{79,121}}]
          R = secondExperiment[{{122,164}}]


          transmittionsX = firstExperiment[{{79,121}}]
          
          if fixNoiseAboveZeroPoint9 then
            transmittionsX = fixNoiseAboveZeroPoint9Func(transmittionsX)
            transmittionsY = fixNoiseAboveZeroPoint9Func(transmittionsY)
          end

          temp = torch.cat(experimentVariables [{{9,33}}],transmittionsY,1) -- no pitch and polarizations - which is 3 paramters.not sure about no pitch. debug it
          temp= torch.cat(temp,transmittionsX,1)
          data[{{foundCounter},{}}] = temp

          labels[{{foundCounter},{}}] = experimentVariables [{{1,8}}]

          table.insert(newFileNames, experiment)
        end

      end
    end
  end

  print "found counter is"
  print(foundCounter) 

  print "among how many experiments?" 
  print (#rawFiles)

  print "removing raw experiments and changing table to new experiments names"
  newData = torch.FloatTensor(foundCounter, rows,columns)
  newLabels = torch.FloatTensor(foundCounter,labelRows)
  newData[{}] = data[{{1,foundCounter},{}}]
  newLabels[{}] = labels[{{1,foundCounter},{}}]
  data = newData
  labels = newLabels


  return data, labels, newFileNames


end




--use it to make direct data consistent with the inverse data
function filterNoPairDirectData(samples, rawFiles, pol)
  print("samples[1].output_data")
  print (samples[1].output_data)
  print("rawFiles[1]")
  print (rawFiles[1])

  print("samples[2].output_data")
  print (samples[2].output_data)
  print("rawFiles[2]")
  print (rawFiles[2])


  print "creating new array to hold the merged-new experiments names"
  local newFileNames = {}
  local newSamples = {}
  


  pattern = "_polarization_"
  patternLen = pattern:len()

  foundCounter = 0
  if illustrate then
    howManyToLoad = batchSize
  else
    howManyToLoad = #rawFiles
  end

  for j= 1,howManyToLoad do
    found = false
    --get x polarization experiment representation - we also check for equality in the experiment representation due to data augmentation 
    --that can lead to two matching experiments with different prefix   
    firstExperiment = samples[j].output_data
    matching = firstExperiment[{{1,35}}]
    matching = torch.Tensor(matching:size()):copy(matching)
    --look for x polarization
    --if string.find(rawFiles[j],"x1y0_P_700.mat") then
    if torch.all(firstExperiment[34]:eq(1))then

      experiment = rawFiles[j]
      startIndex = string.find(experiment,pattern)
      prefix = string.sub(experiment,0,startIndex+patternLen-1)
      secondExperimentName = prefix .. "x0y1_P_700.mat"
      
      --look for the second experiment
      --remember not to use the second experiment during the iterations
      found = false
      matching[34] = 0
      matching[35] = 1.0
      for i = 1,#rawFiles do
        
        if found then  ---------new code to avoid duplication of the same merged experiments when there are two or more matches in the dataset
          break
        end
  
        -- torch.all((x*0):eq(y*0)) --compare two tensors 
        secondExperiment = samples[i].output_data
        experimentVariables = secondExperiment[{{1,35}}]

        --        if (rawFiles[i] == secondExperimentName) or (torch.all(matching:eq(experimentVariables)))  then
        if torch.all(matching:eq(experimentVariables)) then
          found = true
          foundCounter = foundCounter + 2

          if whichPolarization == 1 or whichPolarization == 0 then
          table.insert(newSamples,firstExperiment)
          table.insert(newFileNames, experiment)
          end
          
          if whichPolarization == 2 or whichPolarization == 0 then
            table.insert(newSamples,secondExperiment)
            table.insert(newFileNames, secondExperimentName)
          end
--          i = #rawFiles+1 --double break temp hack
        end

      end
    end
  end

  print "found counter is"
  print(foundCounter) 

  print "among how many experiments?" 
  print (#rawFiles)

  print "removing raw experiments and changing table to new experiments names"
--  newData = torch.FloatTensor(foundCounter, rows,columns)
--  newLabels = torch.FloatTensor(foundCounter,labelRows)
--  newData[{}] = data[{{1,foundCounter},{}}]
--  newLabels[{}] = labels[{{1,foundCounter},{}}]
--  data = newData
--  labels = newLabels


  return newSamples, newFileNames


end


--TODO fix me!!!! 
function inverseNormalizationTranmission(pred, pol)
  if pol ==2 then -- was 1 and 1 is a bug!
    offest = 26
  else
    offest = 69
  end
  for i=offest,offest+42 do --for loop is until i <= offset + 42
    --        print '(pred[1][i])'
    --        print (pred[1][i])
    pred[i-offest+1] = pred[i-offest+1] * trainDataStd[i]
    pred[i-offest+1] = pred[i-offest+1] + trainDataMean[i]

  end
  return pred
end



-------------------------------
-------------------------------
--moved from normalization.lua
-------------------------------
-------------------------------
function inverseNormalizationOnLabelBatch(pred)
--  print(pred)

  for j = 1, pred:size()[1] do
  
  
    for i=1,labelLength do
      --        print '(pred[1][i])'
      --        print (pred[1][i])
      pred[j][i] = pred[j][i] * trainLabelsOriginalStd[i]
      pred[j][i] = pred[j][i] + trainLabelsOriginalMean[i]
  
    end
  end
  return pred
end

function normelizeTestLabelsBatch(data)
  --normelize query according to original train data
  
   for i=1,labelLength do
    data[{ {},{i} }]:add(-trainLabelsOriginalMean[i])--TODO i canceled it
    data[{ {},{i} }]:div(trainLabelsOriginalStd[i])
   end
  
  return data
end

function normelizeTestLabels(data)
  --normelize query according to original train data
   for i=1,labelLength do
      data[{{i}}]:add(-trainLabelsOriginalMean[i])--TODO i canceled it
      data[{{i}}]:div(trainLabelsOriginalStd[i])
  end
  return data
end



function inverseNormalizationOnLabel(pred)
--  print("pred")
--  print(pred)
--  if splittedModel then
--    first =  pred[{1,{1,25}}]
--    second = pred[{1,{}}]
--    third = pred[{1,{}}]
--    temp = torch.cat(first,second,1)
--    pred = torch.cat(temp,third,1)
--  end
-----------------------------------------
--  for i=1,2 do
--    --        print '(pred[1][i])'
--    --        print (pred[1][i])
--    pred[i] = pred[i] * trainLabelsOriginalStd[6+i]
--    pred[i] = pred[i] + trainLabelsOriginalMean[6+i]
--
--  end
-----------------------------------------

  for i=1,labelLength do
    --        print '(pred[1][i])'
    --        print (pred[1][i])
    pred[i] = pred[i] * trainLabelsOriginalStd[i]
    pred[i] = pred[i] + trainLabelsOriginalMean[i]

  end

  return pred
end

----------------------------------------------------------------------