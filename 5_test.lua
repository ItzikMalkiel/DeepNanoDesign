----------------------------------------------------------------------
-- This script implements a test procedure, to report accuracy
-- on the test data. Nothing fancy here...
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
  
  -- local vars
  local time = sys.clock()
  testTotalErr = 0

  if average then
    cachedparams = parameters:clone()
    parameters:copy(average)
  end

  -- set model to evaluate mode (for modules that differ in training and testing, like Dropout)
  model:evaluate()

  -- test over test data
  print('==> testing on test set:')
  for t = 1,data:size()[1] do
    -- disp progress
    xlua.progress(t, data:size()[1])

    -- get new sample
    local input = data[t]
    local target = labels[t]

    input = torch.DoubleTensor(input:size()):copy(input) -- new code may 2018 
    target = torch.DoubleTensor(target:size()):copy(target)
--    print "avoiding cuda..."

    local pred = model:forward(input)
    local err = criterion:forward(pred,target)
    testTotalErr = testTotalErr + err
    target = torch.Tensor(labelLength):copy(target)---------------------------is it a bug?
    

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
      print 'testFiles[t]'
      print (dataFiles[t])
    end 


    if normalizeLabels then
      if splittedModel then
        pred = inverseNormalizationOnLabel(pred)
      else
        pred = inverseNormalizationOnLabel(pred[1])
      end
      target = inverseNormalizationOnLabel(target)
    end

--          print('ground truth')
--          print ( target)
--          print('prediction after inverse normalization')
--          print ( pred)
    if writeDataForStage2 then
      pred = pred:double()
      input = input:double()
      target = torch.Tensor(1,labelLength):copy(target)
      newLabel = target:double() - pred
      if t  < polttingAmount then
        plot(target,pred,newLabel, testFiles[t])
      end
      toWrite = torch.cat(input,newLabel)
    else 
      toWrite = pred
    end

    
    -- turn on if you want to use cascade architecture
    if stage2 then
    
      newPred = criterion:forward(pred,target)
      original = criterion:forward(pred*0,target)
      
      if newPred < original then
      
        if t < polttingAmount then
         plot(target,pred,pred*0-1,testFiles[t])
        end

        numOfBetterPredictions = numOfBetterPredictions + 1
        
      else
        if t < polttingAmount then
         plot(target,pred,pred*0-1,testFiles[t])
        end
      end
      overallImprovmentOfBetterPredictions = overallImprovmentOfBetterPredictions + original - newPred
    end
    
    if t < polttingAmount and not stage2 then
         printDebug(input)  
         printDebug(testFiles[t])       
         plot(target,pred,pred*0,testFiles[t])
    end

    writeBatchResultTo(path, toWrite, dataFiles, t, target)
  end
  
  
  testTotalErr = testTotalErr / data:size()[1]

  if testTotalErr + 0.0000000005 < testTotalMinErr and saveBestNetwork   then
    testTotalMinErr = testTotalErr
    testTotalMinErrString = '' .. testTotalMinErr
    saveNetwork('bestResultOnTest')
  end

  -- timing
  time = sys.clock() - time
  time = time / data:size()[1]
  print("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')

  -- update log/plot
--  testTotalErr = '' .. testTotalErr
  testLogger:add{['epoch: % fSumTest: % fMinTest: %'] = epoch, testTotalErr ..'',testTotalMinErrString}----------------------------------------------
  --   testLogger:add{['Cost function value (test set)'] = testTotalErr}
end
