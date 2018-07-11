--require('mobdebug').start()

print '==> processing options'

print '==> loading utils'
dofile 'utils.lua'

print '==> configuring project...'
dofile 'configuration.lua'
dofile 'buildLossFunction.lua'

if bothDirections then

  dofile 'trainBothDirectionsInverse.lua' -- was this one
  dofile 'testBothDirectionsInverse.lua'
  

else
  dofile 'train.lua'
  dofile 'test.lua'
end


if loadBestTestNetwork == 1 and continueLearning == 0 then
--  dofile '5_test.lua'
  test(testData, testLabels, testFiles, pathToWritePredictions, true, writeDataForStage2)
  exit()
end


i=1
epoch = epoch or 1
if continueLearning == 1 then
  test(testData, testLabels, testFiles, pathToWritePredictions, true)
end
----------------------------------------------------------------------
print '==> training!'
while true do
  train()

  if i % testEveryXEpochs == 0 and i > boost then
    test(testData, testLabels, testFiles, pathToWritePredictions, true)
    
    --create data for stage2
    if writeDataForStage2 then
      dir = 'stage2-' .. i
      os.execute("mkdir " .. dir)
      dir = dir ..'/'
      test(trainData, trainLabels, trainFiles, dir, false, writeDataForStage2)-------fix me if needed 
    end
  end

  print ('new phase epoch #'.. i .. 'is done!')
  if i % 2000000 == 0 then
    --writeTrainDataForStage2()

    -- save/log current net
    saveNetwork(opt.save)
    -- grabage collection after every batch
    collectgarbage()
  end
  
  if i % testEveryXEpochs == 0 then
--    print ('learning rate is: ' .. optimState.learningRate)
--    print ('evalCounter is: ' .. optimState.evalCounter)
  end
  
  i = i + 1
  

end	






