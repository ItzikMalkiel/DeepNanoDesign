----------------------------------------------------------------------
-- This script implements a test procedure, to report accuracy
-- on the test data. Nothing fancy here...
--
-- Clement Farabet
----------------------------------------------------------------------

require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods

----------------------------------------------------------------------
testTotalMinErr = 10000000

print '==> defining test procedure'

-- test function
function test(data, labels, dataFiles, path, saveBestNetwork)

  local numOfBetterPredictions = 0
  local overallImprovmentOfBetterPredictions = 0
  local totalScore = 0
  local totalSuccess = 0

  local time = sys.clock()
  testTotalErr = 0
  testTotalErrX = 0 
  testTotalErrY = 0
  testTotalErrInverse = 0

  bothNetworks:evaluate()

  print("data:size()")
  print(data:size())
  print("labels:size()")
  print(labels:size())

  -- test over test data
  print('==> testing on test set:')
  for t = 1,data:size()[1] do
    -- disp progress
    xlua.progress(t, data:size()[1])

    -- get new sample
    local input = data[t]
    local target = labels[t]

    --    input = input:cuda()
    --    target = target:cuda()

    local inverseData = buildInverseDataNoBatch(input[{3,{}}], input[{2,{}}], input)

    inverseData = inverseData:cuda()
    local outputInverse = modelInverse:forward(inverseData)
    local errInverse = criterion:forward(outputInverse, target:cuda()) --compare to geometries
    targetTmp = target:clone()
    targetTmp = inverseNormalizationOnLabel(targetTmp)
    outputInverse = inverseNormalizationOnLabel(outputInverse)
    outputInverse = torch.round(outputInverse)
    local geometry = outputInverse:clone()
    print (outputInverse)
    print(targetTmp)
    outputInverse = normelizeTestLabels(outputInverse)

    directX, targetsX = buildDirectDataNoBatch(input, outputInverse, 1)
    directY, targetsY = buildDirectDataNoBatch(input, outputInverse, 2)

    directX = directX:cuda()
    targetsX = targetsX:cuda()
    directY = directY:cuda()
    targetsY = targetsY:cuda()


    if modelXY_enabled then 

      outputX = modelXY:forward(directX):clone()
      errX = criterion:forward(outputX, targetsX) 

      outputY = modelXY:forward(directY):clone()
      errY = criterion:forward(outputY, targetsY) 
    else
      local outputX = modelX:forward(directX):clone()
      errX = criterion:forward(outputX, targetsX) 

      local outputY = modelY:forward(directY):clone()
      errY = criterion:forward(outputY, targetsY) 

      outputX = outputX:float()
      outputY = outputY:float()

    end

    local err = errX + errY  + errInverse -- + errX + errY -------was without the last two
    testTotalErr = testTotalErr + (err/3)
    testTotalErrX = testTotalErrX + errX
    testTotalErrY = testTotalErrY + errY
    testTotalErrInverse = testTotalErrInverse + errInverse

    pred = outputInverse

    target = torch.Tensor(target:size()):copy(target)---------------------------is it a bug?

    local score = 0
    targetsX = inverseNormalizationTranmission(targetsX:clone(),1)
    targetsY = inverseNormalizationTranmission(targetsY:clone(),2)
    outputX = inverseNormalizationTranmission(outputX:clone(),1)
    outputY = inverseNormalizationTranmission(outputY:clone(),2)
    for index=1,43 do
      if torch.abs(outputX[index] - targetsX[index]) < 0.1 then
        score = score + 1
      end

      if torch.abs(outputY[index] - targetsY[index]) < 0.1  then
        score = score + 1
      end
    end
    if score/(43+43) > 0.92 then
      totalSuccess = totalSuccess + 1
    end
    totalScore = totalScore + score/(43+43)

    if (err > 100) then
      print('before adding mean and std')
      print 'pred'
      print ( pred)
      print 'labelLength'
      print ( labelLength)
      print 'groundTruth'
      print ( target)
      print 'local err'
      print ( err)
      print 'testTotalErr'
      print ( testTotalErr)
    end 

    if t < polttingAmount and not stage2 then
--      targetsX = inverseNormalizationTranmission(targetsX:clone(),1)
--      targetsY = inverseNormalizationTranmission(targetsY:clone(),2)
--      outputX = inverseNormalizationTranmission(outputX:clone(),1)
--      outputY = inverseNormalizationTranmission(outputY:clone(),2)

      printDebug(input)  
      printDebug(testFiles[t])       
      plot(targetsX,outputX,outputX*0,testFiles[t].. '_x_' .. geometry[1] .. '-' .. geometry[2] .. '-' .. geometry[3] .. '-' .. geometry[4] .. '-' .. geometry[5] .. '-' .. geometry[6] .. '-' .. geometry[7] .. '-' .. geometry[8])
      plot(targetsY,outputY,outputY*0,testFiles[t].. '_y_' .. geometry[1] .. '-' .. geometry[2] .. '-' .. geometry[3] .. '-' .. geometry[4] .. '-' .. geometry[5] .. '-' .. geometry[6] .. '-' .. geometry[7] .. '-' .. geometry[8])
      print ('testFiles[t]')
      print (testFiles[t])
      print ('outputInverse')
      print (outputInverse)
    end

  end

  print('we have improved ' .. numOfBetterPredictions .. ' predictions!' )
  print('we have improved them by abs of: ' .. overallImprovmentOfBetterPredictions )
  waitDebug(5)

  testTotalErr = testTotalErr / data:size()[1]
  testTotalErrX = testTotalErrX / data:size()[1]
  testTotalErrY = testTotalErrY / data:size()[1]
  testTotalErrInverse = testTotalErrInverse / data:size()[1]


  if testTotalErr + 0.0000000005 < testTotalMinErr and saveBestNetwork    then
    testTotalMinErr = testTotalErr
    testTotalMinErrString = '' .. testTotalMinErr
    saveNetwork('bestResultOnTest')
  end

  -- timing
  time = sys.clock() - time
  time = time / data:size()[1]
  print("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')

  -- update log/plot
  testLogger:add{['epoch: % fSumTest: % fMinTest: % totalErrX: % totalErrY: % totalErrInverse: %'] = epoch, testTotalErr ..'',testTotalMinErrString
    ,testTotalErrX, testTotalErrY, testTotalErrInverse }----------------------------------------------

  totalScore = totalScore / data:size()[1]
  totalSuccess = totalSuccess / data:size()[1] 
  print ("TOTAL SCORE IS: " .. totalScore .. " !")
  print ("TOTAL SUCCESS IS: " .. totalSuccess .. " !")

end
