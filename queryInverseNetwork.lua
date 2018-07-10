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
plotting = true
roundGeometryPrediction = true

function buildGrpah(resonance, width, depth)
  local trans = torch.FloatTensor(43)
  for i=1,43 do
    if resonance - i > width then  --we are far from the right side of the resonance
      trans[i] = 0.95
    elseif torch.abs(resonance - i) < width then --we are in the right side of the resonance 
      trans[i] = 1 -  depth  * torch.pow((1-(torch.abs(resonance - i )/width)),2)
    else 
      trans[i] = 0.95
    end
  end
  
  return trans
end



print "==>load test file"
filename = 'new_mat_Au_epsDiele_1.2_geom_5_lmbd_600_25_1650_L0_230_L1_210_h_40_w_40_theta_0_polarization_x1y0_P_700.mat'
filePath =  'dataset/datasetVersion5.5/test/' ..filename
rawGroundTruth = mattorch.load(filePath)
transXRaw = rawGroundTruth.output_data[{{79,121}}]:t()[1]:float()

--new_mat_Au_epsDiele_1.2_geom_6_lmbd_600_25_1650_L0_247_L1_218_h_40_w_56_theta_0_polarization_x0y1_P_700 --interesting fabrication!
filename = 'new_mat_Au_epsDiele_1.2_geom_5_lmbd_600_25_1650_L0_230_L1_210_h_40_w_40_theta_0_polarization_x0y1_P_700.mat'
filePath =  'dataset/datasetVersion5.5/test/' ..filename
rawGroundTruth = mattorch.load(filePath)
transYRaw = rawGroundTruth.output_data[{{79,121}}]:t()[1]:float()


--transXRaw = torch.pow(transXRaw, (600)/(700))
--transYRaw = torch.pow(transYRaw, (600)/(700))


directRaw =  {
  --geometry
  1,1,0,0,200000000, 1,230, 0, 
  --material
  9.0300,    0.7700,    0.0500,    0.0540,    0.0740,    0.2180,    0.7420,    0.0500,
  0.0350,    2.8850,   0.3490,    0.3120,    0.0830,    4.0690,    0.8300,    0.7190,
  0.1250,    6.1370,    1.2460,    1.6480,    0.1790,   27.9700,    1.7950,  700.0000,
  --eps and polarization
  1.2,1,0} 

transXRaw1 = {0.99,  --600 
  0.98,        --625
  0.97,        --625
  0.96,        --625
  0.95,        --700
  0.94,        --625
  0.93,        --
  0.92,        --
  0.91,        --800
  0.9,        --
  0.9,        --
  0.9,        --
  0.9,        --900
  0.9,        --
  0.9,        --
  0.9,        --
  0.9,        --1000
  0.9,        --
  0.9,        --
  0.9,        --
  0.85,        --1100
  0.8,        --
  0.7,        --
  0.6,        --
  0.7,        --1200
  0.8,        --
  0.85,        --
  0.9,        --
  0.9,        --1300
  0.9,        --
  0.9,        --
  0.9,        --
  0.9,        --1400
  0.9,        --1425
  0.9,        --1450
  0.9,        --1475
  0.9,        --1500
  0.9,        --1525
  0.9,        --1550
  0.9,        --1575
  0.9,        --1600
  0.9,        --1625
  0.9         --1650
}

transYRaw1 = {0.9606,  --600 
  0.9432,        --625
  0.9732,        --625
  0.9819,        --625
  0.9928,        --700
  0.9957,        --625
  0.9,        --
  0.85,        --
  0.8,        --800
  0.73,        --
  0.8,        --
  0.85,        --
  0.9,        --900
  0.9244,        --
  0.98509,        --
  0.97193,        --
  0.97322,        --1000
  0.98548,        --
  0.99159,        --
  0.99409,        --
  0.9516,        --1100
  0.9557,        --
  0.9561,        --
  0.9536,        --
  0.9498,        --1200
  0.9452,        --
  0.9397,        --
  0.9329,        --
  0.9248,        --1300
  0.9152,        --
  0.9040,        --
  0.98911,        --
  0.98753,        --1400
  0.98560,        --1425
  0.98326,        --1450
  0.98043,        --1475
  0.97698,        --1500
  0.97279,        --1525
  0.96772,        --1550
  0.96164,        --1575
  0.95453,        --1600
  0.94649,        --1625
  0.93796         --1650
}


direct = torch.FloatTensor(directRaw) 
transX = torch.FloatTensor(transXRaw)
transY = torch.FloatTensor(transYRaw)




--build waveLengths
waveLengths = torch.FloatTensor(43)
for i=1,43 do
  waveLengths[i] = 575 +i*25
end
print(waveLengths)

plot(transX,transY,transX*0,'x-y')

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
--  print ("there might be a bug - think twice when you use this function... exiting!")
--  exit()
  for i=1,labelLength do
    --        print '(pred[1][i])'
    --        print (pred[1][i])
    pred[i] = pred[i] - trainLabelsOriginalMean[i]
    pred[i] = pred[i] / trainLabelsOriginalStd[i]

  end
  return pred
end

function inverseNormalizationTranmission(pred, pol)
  if pol ==1 then
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

function normelizeTestData(data)
  --normelize query according to original train data
  for i=1,numOfFeatures do
    data[1][i] = data[1][i] -trainDataMean[i]
    data[1][i] = data[1][i] / trainDataStd[i]
  end
end

function queryInverse(input,label, modelArg)
  rawOutput = modelArg:forward(input:cuda())

  if PitchTransform then
    output = torch.pow(output, (DLPitch)/(labelPitch))
    --    output = torch.pow(output, (DLPitch*DLPitch)/(labelPitch*labelPitch))
  end
  --  outputAfterNormalization = inverseNormalizationOnLabel(output)
  plotting = true
  return rawOutput
    --  plot(label,output,output*0,filename)
end



print '==> processing options'

print '==> loading normalization values'
trainLabelsOriginalStd = torch.load('normalization/direct-trainLabelsStd' )
trainLabelsOriginalMean = torch.load('normalization/direct-trainLabelsMean')
trainDataStd = torch.load('normalization/direct-trainDataStd')
trainDataMean = torch.load('normalization/direct-trainDataMean')

print '==> loading ground truth'


DLPitch = 700
labelPitch = 300
PitchTransform = false

print('==>loading network')
model = torch.load('bestResultOnTest/pretrained/model-inverse-5.5.net') ---------TODO fix me if needed
model:evaluate()
model:cuda()


print '==> normelize test data'
testData = torch.FloatTensor(1,43+43+25)
temp = torch.cat(direct[{{1+8,25+8}}]:float(), transY, 1)
testData[1] = torch.cat(temp, transX:float(), 1)
numOfFeatures = 111
labelLength = 8
normelizeTestData(testData)



print '==> build inverse data'
inverseData = torch.FloatTensor(3,43)
print (direct[{{1+8,25+8}}])
inverseData[1] =  torch.cat(testData[1][{{1,25}}]:float(), torch.zeros(18):float(),1)
inverseData[2] = testData[1][{{26,26+42}}]
inverseData[3] = testData[1][{{26+43,26+43+42}}]

print('inverseData')
print(inverseData)
geometryRaw = queryInverse(inverseData,groundTruth, model)
geometry = inverseNormalizationOnLabel(geometryRaw:clone())

print ("geometry 1")
print (geometry)

print ("temp")
print (temp)


if roundGeometryPrediction then
temp = torch.round(geometry)
print ("geometry 2 round")
print (temp)
end

print ("ground truth")
print (target)








































--function writeBatchResultTo(path, prediction, filenames, index, groundTruth)
--print("do nothing")
--end
--
--print '==> processing options'
--
--print '==> loading utils'
--dofile 'utils.lua'
--
--
--print '==> configuring project...'
--dofile 'configuration.lua'
--
--
--
--print('==>loading network')
--model = torch.load('bestResultOnTest/model.net')
-- 
--dofile '3_loss.lua'
--
--input = torch.FloatTensor(1, numOfFeatures)
----case 6
----queryTable =  {1.0000,   1.0000,   0.0000,   0.0000, 140.0000,   1.0000, 120.0000,   29.0000, 
---- 14.9800,   0.5260,   0.0470,   0.2130,   0.3120,   0.1630,   0.0130,   0.0600,   0.3150,
---- 1.5610,   0.0420,   0.1820,   1.5870,   1.8270,   0.2560,   0.0140,   2.1450,   4.4950, 1.7350,
----   0.0000,   0.0000,   0.0000,   0.0000, 700.0000,   1.0000,   1.0000,   0.0000}
--   
----fixed_mat_Au_epsDiele_3.8_geom_6tag2_lmbd_600_25_1650_L0_230_L1_210_h_40_w_40_thata_90_polarization_x1y0.mat 
----  queryTable =  {0, 0,  1, 0,  210,  1,  400, 0, 
----    9.0300,    0.7700,    0.0500,    0.0540,    0.0740,    0.2180,    0.7420,    0.0500,
----   0.0350,    2.8850,   0.3490,    0.3120,    0.0830,    4.0690,    0.8300,    0.7190,
----   0.1250,    6.1370,    1.2460,    1.6480,    0.1790,   27.9700,    1.7950,  700.0000,
----   1.0000,    1.0000,    0.0000} 
--   
----fixed_mat_Au_epsDiele_2.25_geom_6_lmbd_600_25_1650_L0_185_L1_165_h_40_w_40_thata_80_polarization_x1y0.mat
---- queryTable =  {1, 1,  0, 0,  165,  1,  185, 80, 
----    9.0300,    0.7700,    0.0500,    0.0540,    0.0740,    0.2180,    0.7420,    0.0500,
----   0.0350,    2.8850,   0.3490,    0.3120,    0.0830,    4.0690,    0.8300,    0.7190,
----   0.1250,    6.1370,    1.2460,    1.6480,    0.1790,   27.9700,    1.7950,  700.0000,
----   1.0000,    1.0000,    0.0000} 
--   
----my test   
--queryTable =  {1, 1,  0, 0,  200,  1,  140, 29, 
--    9.0300,    0.7700,    0.0500,    0.0540,    0.0740,    0.2180,    0.7420,    0.0500,
--   0.0350,    2.8850,   0.3490,    0.3120,    0.0830,    4.0690,    0.8300,    0.7190,
--   0.1250,    6.1370,    1.2460,    1.6480,    0.1790,   27.9700,    1.7950,  700.0000,
--   1.0000,    0.0000,    1.0000} 
--
--query = torch.FloatTensor(queryTable) 
--input[1] = query  
----normelize query according to original train data
--for i=1,numOfFeatures do
--  input[1][i] = query[i] -trainDataMean[i]
--  input[1][i] = input[1][i] / trainDataStd[i]
--end
--model:evaluate()
--model:cuda()
--  
--function queryModel(input,label)
--  rawOutput = model:forward(input:cuda())
--  output = inverseNormalizationOnLabel(rawOutput[1])
--  inverseNormalizationOnLabel(label)
--  plotting = true
--  plot(label,output,output*0,'good')
--end
--queryModel(input,testLabels[30])
--
--
--
----[[
--    0.0000
--    0.0000
--    1.0000
--    0.0000
--  210.0000
--    1.0000
--  400.0000
--    0.0000
--    9.0300
--    0.7700
--    0.0500
--    0.0540
--    0.0740
--    0.2180
--    0.7420
--    0.0500
--    0.0350
--    2.8850
--    0.3490
--    0.3120
--    0.0830
--    4.0690
--    0.8300
--    0.7190
--    0.1250
--    6.1370
--    1.2460
--    1.6480
--    0.1790
--   27.9700
--    1.7950
--  700.0000
--    1.0000
--    1.0000
--    0.0000
--  600.0000 
--  625.0000
--  650.0000
--  675.0000
--  700.0000
--  725.0000
--  750.0000
--  775.0000
--  800.0000
--  825.0000
--  850.0000
--  875.0000
--  900.0000
--  925.0000
--  950.0000
--  975.0000
-- 1000.0000
-- 1025.0000
-- 1050.0000
-- 1075.0000
-- 1100.0000
-- 1125.0000
-- 1150.0000
-- 1175.0000
-- 1200.0000
-- 1225.0000
-- 1250.0000
-- 1275.0000
-- 1300.0000
-- 1325.0000
-- 1350.0000
-- 1375.0000
-- 1400.0000
-- 1425.0000
-- 1450.0000
-- 1475.0000
-- 1500.0000
-- 1525.0000
-- 1550.0000
-- 1575.0000
-- 1600.0000
-- 1625.0000
-- 1650.0000
--    0.9606
--    0.9432
--    0.9732
--    0.9819
--    0.9928
--    0.9957
--    0.9851
--    0.9888
--    0.9878
--    0.9845
--    0.9796
--    0.9718
--    0.9566
--    0.9244
--    0.8509
--    0.7193
--    0.7322
--    0.8548
--    0.9159
--    0.9409
--    0.9516
--    0.9557
--    0.9561
--    0.9536
--    0.9498
--    0.9452
--    0.9397
--    0.9329
--    0.9248
--    0.9152
--    0.9040
--    0.8911
--    0.8753
--    0.8560
--    0.8326
--    0.8043
--    0.7698
--    0.7279
--    0.6772
--    0.6164
--    0.5453
--    0.4649
--    0.3796
--    0.0314
--    0.0316
--    0.0242
--    0.0332
--    0.0175
--    0.0179
--    0.0200
--    0.0192
--    0.0198
--    0.0206
--    0.0219
--    0.0247
--    0.0303
--    0.0419
--    0.0675
--    0.1099
--    0.0990
--    0.0560
--    0.0361
--    0.0286
--    0.0260
--    0.0257
--    0.0265
--    0.0281
--    0.0303
--    0.0330
--    0.0363
--    0.0401
--    0.0447
--    0.0500
--    0.0563
--    0.0639
--    0.0730
--    0.0839
--    0.0971
--    0.1132
--    0.1328
--    0.1567
--    0.1857
--    0.2205
--    0.2614
--    0.3075
--    0.3567
----]]

