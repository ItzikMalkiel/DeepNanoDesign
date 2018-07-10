require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods

----------------------------------------------------------------------
print '==> changing model to cuda'

--modelX:cuda()
--modelY:cuda()
modelInverse:cuda()
bothNetworks:cuda()
criterion:cuda()

--modelXY:cuda() ---------------TODO new code

----------------------------------------------------------------------
print '==> defining some tools'

-- Log results to files
trainLogger = optim.Logger(paths.concat('results', 'train.log'))
testLogger = optim.Logger(paths.concat('results', 'test.log'))

bothNetworksParameters,bothNetworksGradParameters = bothNetworks:getParameters()

function buildInverseData(outputX, outputY, inputs)
  local numOfExperiments = inputs:size()[1]
  local direct = torch.FloatTensor(numOfExperiments, columns,rows)
  local transmission = torch.FloatTensor(numOfExperiments, 8)
  local newData = torch.FloatTensor(numOfExperiments,3,43)

  print ('numOfExperiments')
  print (numOfExperiments)
  print ('outputY:size()')
  print (outputY:size())

  for i=1,numOfExperiments do
    newData[{i,1,{}}] =  torch.cat(inputs[{i,1,{1,25}}], torch.zeros(18):float(),1)
    --    newData[{i,1,{}}] =  inputs[{i,1,{}}]
    newData[{i,2,{}}] = outputY[{i}]:float()
    newData[{i,3,{}}] = outputX[{i}]:float()
    --    newData[{i,3,{}}] = outputX[{i,{}}]:float()
  end

  return newData
end

function getTheGeometries(inputs)
  return inputs[{{},1,{1,8}}]
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

function getTheGeometriesNoBatch(inputs)
  return inputs[{1,{1,8}}]
end

----------------------------------------------------------------------



print '==> defining training procedure'

function train()

  local time = sys.clock()
  bothNetworks:training()

  -- shuffle at each epoch
  shuffle = torch.randperm(trainData:size()[1]):long()
  trainDataLocal = trainData:index(1, shuffle)
  trainLabelsLocal = trainLabels:index(1, shuffle)

  -- do one epoch
  print('==> doing epoch on training data:')
  print('==> learningRate = ' .. optimState.learningRate)
  print("==> online epoch # " .. epoch .. ' [batchSize = ' .. batchSize .. ']')
  totalErr = 0
  totalErrX = 0
  totalErrY = 0
  totalErrInverse= 0
  for t = 1,trainDataLocal:size()[1],batchSize do
    -- disp progress
    xlua.progress(t, trainDataLocal:size()[1])

    -- create mini batch
    local lastIndex = math.min(t+batchSize-1,trainDataLocal:size()[1])
    local inputs = trainDataLocal[{{t,lastIndex}}]:clone()
    local targets = trainLabelsLocal[{{t,lastIndex}}]:clone()

    -- create closure to evaluate f(X) and df/dX
    local feval = function(x)

      bothNetworksGradParameters:zero()
      local inverseData = buildInverseData(inputs[{{},3,{}}], inputs[{{},2,{}}], inputs)
      inverseData = inverseData:cuda()
      local outputInverse = modelInverse:forward(inverseData)
      local errInverse = criterion:forward(outputInverse, targets:cuda()) --compare to geometries
      if epoch % backwardInverseEvery == 0  then
        local df_do_inverse = criterion:backward(outputInverse, targets:cuda())  
        modelInverse:backward(inverseData, df_do_inverse)
      end

      outputInverse = inverseNormalizationOnLabelBatch(outputInverse)
      outputInverse = torch.round(outputInverse)
      outputInverse = normelizeTestLabelsBatch(outputInverse)
      
      local directX, targetsX = buildDirectData(inputs, outputInverse, 1)
      local directY, targetsY = buildDirectData(inputs, outputInverse, 2)

      if modelXY_enabled then 
      
        local directXY = torch.cat(  directY,directX, 1)
        local targetsXY = torch.cat( targetsY,targetsX, 1)

        directXY = directXY:cuda()
        targetsXY = targetsXY:cuda()

        local outputXY = modelXY:forward(directXY):clone()
        errXY = criterion:forward(outputXY, targetsXY) 
        if i > backwardDirectAfterXEpochs then
          local df_do_directXY = criterion:backward(outputXY, targetsXY)
          modelXY:backward(directXY, df_do_directXY)
        end
        
        local err = errXY*2 + errInverse
        totalErr = totalErr + (err/3)
        totalErrX = totalErrX + errXY 
        totalErrY = totalErrY + errXY
        totalErrInverse = totalErrInverse + errInverse
      
      else

        directX = directX:cuda()
        targetsX = targetsX:cuda()
        directY = directY:cuda()
        targetsY = targetsY:cuda()
  
  
        local outputX = modelX:forward(directX):clone()
        errX = criterion:forward(outputX, targetsX) 
        if i > boost then
          local df_do_directX = criterion:backward(outputX, targetsX)
          modelX:backward(directX, df_do_directX)
        end
  
        local outputY = modelY:forward(directY):clone()
        errY = criterion:forward(outputY, targetsY) 
        if i > boost then
          local df_do_directY = criterion:backward(outputY, targetsY)
          modelY:backward(directY, df_do_directY)
        end
  
        outputX = outputX:float()
        outputY = outputY:float()
  
  
        local err = errX + errY  + errInverse -- + errX + errY -------was without the last two
        totalErr = totalErr + (err/3)
        totalErrX = totalErrX + errX 
        totalErrY = totalErrY + errY
        totalErrInverse = totalErrInverse + errInverse
      
      end

      if t  < polttingAmount and epoch % trainPlotEveryXEpochs == 0 and plotting then
        targetToPlot = inverseNormalizationOnLabel(targets[1]:double())
        outputToPlot = inverseNormalizationOnLabel(output[1]:double())
        plot(targetToPlot,outputToPlot, outputToPlot*0-1)
      end

      return err,bothNetworksGradParameters   ----TODO was not commeneted
    end

    optimMethod(feval, bothNetworksParameters, optimState)
  end

  -- time taken
  time = sys.clock() - time
  time = time / trainData:size()[1]
  print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')

  if criterion.sizeAverage then
    totalErr = totalErr * batchSize / trainData:size()[1]  
    totalErrX = totalErrX * batchSize / trainData:size()[1]  
    totalErrY = totalErrY * batchSize / trainData:size()[1]  
    totalErrInverse = totalErrInverse * batchSize / trainData:size()[1]  

  else
    totalErr = totalErr / trainData:size()[1]
    totalErrX = totalErrX / trainData:size()[1]
    totalErrY = totalErrY / trainData:size()[1]
    totalErrInverse = totalErrInverse / trainData:size()[1]
  end 
  trainLogger:add{['Cost function value: % totalErrX: % totalErrY: % totalErrInverse: %' ] = totalErr, totalErrX, totalErrY, totalErrInverse}

  epoch = epoch + 1
end
