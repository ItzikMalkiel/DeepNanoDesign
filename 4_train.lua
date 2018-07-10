require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods

----------------------------------------------------------------------
print '==> changing model to cuda'
print(model)
--model:cuda()
--criterion:cuda()

----------------------------------------------------------------------
print '==> defining some tools'

-- Log results to files
trainLogger = optim.Logger(paths.concat('results', 'train.log'))
testLogger = optim.Logger(paths.concat('results', 'test.log'))

-- Retrieve parameters and gradients:
-- this extracts and flattens all the trainable parameters of the mode
-- into a 1-dim vector
if model then
   parameters,gradParameters = model:getParameters()
end
----------------------------------------------------------------------
print '==> configuring optimizer'
print '==> defining training procedure'

function train()

   -- epoch tracker
   epoch = epoch or 1

   -- local vars
   local time = sys.clock()

   -- set model to training mode (for modules that differ in training and testing, like Dropout)
   model:training()

   -- shuffle at each epoch
   shuffle = torch.randperm(trainData:size()[1]):long()
   trainDataLocal = trainData:index(1, shuffle)
   trainLabelsLocal = trainLabels:index(1, shuffle)

   -- do one epoch
--   print('==> doing epoch on training data:')
--   print('==> learningRate = ' .. optimState.learningRate)
   print("==> online epoch # " .. epoch .. ' [batchSize = ' .. batchSize .. ']')
   totalErr = 0
   for t = 1,trainDataLocal:size()[1],batchSize do
      -- disp progress
      xlua.progress(t, trainDataLocal:size()[1])

      -- create mini batch
      local lastIndex = math.min(t+batchSize-1,trainDataLocal:size()[1])
      local inputs = trainDataLocal[{{t,lastIndex}}]:clone()
      local targets = trainLabelsLocal[{{t,lastIndex}}]:clone()
      
      inputs = torch.DoubleTensor(inputs:size()):copy(inputs)
      targets = torch.DoubleTensor(targets:size()):copy(targets)
      
      -- create closure to evaluate f(X) and df/dX
      local feval = function(x)
                       -- get new parameters
                       if x ~= parameters then
                          parameters:copy(x)
                       end

                       -- reset gradients
                       gradParameters:zero()

                       -- forward
                       local output = model:forward(inputs)
                       
                       local err = criterion:forward(output, targets) 
                       printDebug ("err is " .. err)
--                       print ("ouput std is " .. output:std())
                       
                       printDebug ("new err is " .. err)
                       printDebug("std prints...")
                       waitDebug(3)
                       
                       totalErr = totalErr + err

                       -- backward 
                       local df_do = criterion:backward(output, targets)
                       ---new code
                       printDebug(output)
                       printDebug(targets)
                       printDebug("df_do")
                       printDebug(df_do)
                       printDebug("df prints...")
                       waitDebug(3)
                       
--                       print("output:mean()")
--                       print(output:mean())
--                       print("output:std()")
--                       print(output:std())
                       
                       printDebug (df_do) 
                       printDebug ("new df_do print ")
                       waitDebug(3)
                       
                       model:backward(inputs, df_do)
                       
                       if t  < polttingAmount and epoch % trainPlotEveryXEpochs == 0 then
                         targetToPlot = inverseNormalizationOnLabel(targets[1]:double())
                         outputToPlot = inverseNormalizationOnLabel(output[1]:double())
                         plot(targetToPlot,outputToPlot, outputToPlot*0-1)
                       end
                       
                       return err,gradParameters
                    end

      -- optimize on current mini-batch
      optimMethod(feval, parameters, optimState)
   end

   -- time taken
   time = sys.clock() - time
   time = time / trainData:size()[1]
   print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')

   -- update logger/plot
   --totalErr = totalErr / torch.ceil(trainData:size()[1] / batchSize)
   if criterion.sizeAverage then
    totalErr = totalErr * batchSize / trainData:size()[1]  
   else
    totalErr = totalErr / trainData:size()[1]
   end 
   trainLogger:add{['Cost function value'] = totalErr}
   
   -- next epoch
   epoch = epoch + 1
end
