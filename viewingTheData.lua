require 'mattorch'

print '==> loading utils'
dofile 'utils.lua'

function initData(data, samples)
  print '==> first experiment for example'

  print (samples[1].x)

  for i = 1,#samples do
    y = samples[i].x
    data[{{i},{}}] = y[1]
  end
end


print '==> load train data'
predictionsSamples = {}
predictionsFiles = {}
loadDataFromPath('predictions/', predictionsSamples, predictionsFiles )



print '==> init train data'
--columns = 1
rows = 8
--data = torch.FloatTensor(#predictionsSamples, columns,rows)
data = torch.FloatTensor(#predictionsSamples,rows)
initData(data, predictionsSamples)
--data = data:transpose(2,3)
--data[{ {},{},2 }]

