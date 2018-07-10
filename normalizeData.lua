print '==> preprocessing train data: normalize train data (mean + std)'
print ' before: '  
print(trainData[1])
trainDataMean = {}
trainDataStd = {}
for i=1,numOfFeatures do
  print('before normalize train data '..i..' feature, mean: ' .. trainData[{ {},{},{i} }]:mean())
  --  print('before, '..i..'-channel, std: ' .. imagesTensor[{ {},i,{},{} }]:std())
  trainDataMean[i] = trainData[{ {},{},{i} }]:mean()
  trainDataStd[i] =  trainData[{ {},{},{i} }]:std()
  print('trainDataMean[i]: ' .. trainDataMean[i])
  print('trainDataStd[i]: ' .. trainDataStd[i])

  if trainDataStd[i] == 0 then 
    trainDataStd[i] = 1
  end  
  trainData[{ {},{},{i} }]:add(-trainDataMean[i])
  trainData[{ {},{},{i} }]:div(trainDataStd[i])

  print('after normalize train data '..i..' feature, mean: ' .. trainData[{ {},{},{i} }]:mean())
end
print ' after: '  
print(trainData[1])


print '==> preprocessing testData: Normalize test data, using the training means/stds'  --TODO might be needed?
print ' before: '  
print(testData[1])
for i=1,numOfFeatures do
  testData[{ {},{},{i} }]:add(-trainDataMean[i])
  testData[{ {},{},{i} }]:div(trainDataStd[i])
end
print ' after: '  
print(testData[1])

if normalizeLabels then
  print '==> preprocessing train labels: Normalize train labels (mean + std)'  --TODO might be needed?
  trainLabelsOriginalMean = {}
  trainLabelsOriginalStd = {}
  print ' before: '  
  print(trainLabels[1])
  labelLength = trainLabels:size(2)
  for i=1,labelLength do

    print('before normalize train label '..i..' feature, mean: ' .. trainLabels[{ {},{i} }]:mean())
    --  print('before, '..i..'-channel, std: ' .. imagesTensor[{ {},i,{},{} }]:std())
    trainLabelsOriginalMean[i] = trainLabels[{ {},{i} }]:mean()
    trainLabelsOriginalStd[i] =  trainLabels[{ {},{i} }]:std()
    print('trainLabelsOriginalMean[i]: ' .. trainLabelsOriginalMean[i])
    print('trainLabelsOriginalStd[i]: ' .. trainLabelsOriginalStd[i])

    if trainLabelsOriginalStd[i] == 0 then 
      trainLabelsOriginalStd[i] = 1
    end  
    trainLabels[{ {},{i} }]:add(-trainLabelsOriginalMean[i])
    trainLabels[{ {},{i} }]:div(trainLabelsOriginalStd[i])
    print('after normalize train label '..i..' feature, mean: ' .. trainLabels[{ {},{i} }]:mean())
  end

  print ' after: '  
  print(trainLabels[1])



  print '==> next step in not needed - only for debug mode.'
  print '==> preprocessing test Labels: Normalize test labels, using the training means/stds'  --TODO might be not needed?
  print ' before: '  
  print(testLabels[1])
  for i=1,labelLength do
    testLabels[{ {},{i} }]:add(-trainLabelsOriginalMean[i])--TODO i canceled it
    testLabels[{ {},{i} }]:div(trainLabelsOriginalStd[i])
  end
  print ' after: '  
  print(testLabels[1])

end


print '==> verify statistics: data verification'

for i=1,numOfFeatures do
  trainMean = trainData[{ {},{},{i} }]:mean()
  trainStd = trainData[{ {},{},{i} }]:std()

  testMean = testData[{ {},{},{i} }]:mean()
  testStd = testData[{ {},{},{i} }]:std()

  print('training data, '..i..'-channel, mean: ' .. trainMean)
  print('training data, '..i..'-channel, standard deviation: ' .. trainStd)

  print('test data, '..i..'-channel, mean: ' .. testMean)
  print('test data, '..i..'-channel, standard deviation: ' .. testStd)
end

print '==> verify statistics: train labels verification'
print '==> ================train labels ========================='


trainLabelsMean = trainLabels[{ {},{} }]:mean()
trainLabelsStd = trainLabels[{ {},{} }]:std()

print('trainLabelsMean: ' .. trainLabelsMean)
print('trainLabelsStd: ' .. trainLabelsStd)

print '==> ========================= test labels =================================='
testLabelsMean = testLabels[{ {},{} }]:mean()
testLabelsStd = testLabels[{ {},{} }]:std()

print('testLabelsMean: ' .. testLabelsMean)
print('testLabelsStd: ' .. testLabelsStd)

if normalizeLabels then
  print '==> ========================= mean and std on labels =================================='

  print('trainLabelsMean (for reverse function - original values): ' .. trainLabelsOriginalMean[1])
  print('trainLabelsStd: (for reverse function - original values): ' .. trainLabelsOriginalStd[1])
end

print('==> important: normalizeLabels is ')
print(normalizeLabels)
wait(2)

print '==> ========================= finished normelizing targets =================================='

print '==> printing first train data'
print (trainData[1])
print '==> printing first train label'
print (trainLabels[1])
print '==> printing first test data'
print (testData[1])
print '==> printing first test label'
print (testLabels[1])


if saveNormalizationParams then
  print 'saving normalization params to normalizations/'
  --save train data normalization values
  torch.save('normalization/direct-trainDataStd', trainDataStd)
  torch.save('normalization/direct-trainDataMean', trainDataMean)
  
  --save labels normalization values
  torch.save('normalization/direct-trainLabelsStd', trainLabelsOriginalStd)
  torch.save('normalization/direct-trainLabelsMean', trainLabelsOriginalMean)
end

