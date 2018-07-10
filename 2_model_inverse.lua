
function build(model, noutputs, ninputs)

print('building model....')
scalar = 5
layerSize = 800

model:add(nn.Reshape(ninputs))
model:add(nn.Linear(ninputs,layerSize))
model:add(nn.ReLU())
model:add(nn.Dropout(0.1)) -- used it was 0.5

model:add(nn.Linear(layerSize,layerSize))
model:add(nn.ReLU())
model:add(nn.Dropout(0.1)) -- used it was 0.5

model:add(nn.Linear(layerSize,layerSize))
model:add(nn.ReLU())
model:add(nn.Dropout(0.1)) -- used it was 0.5

model:add(nn.Linear(layerSize,layerSize))
model:add(nn.ReLU())
model:add(nn.Dropout(0.1)) -- used it was 0.5

model:add(nn.Linear(layerSize,layerSize))
model:add(nn.ReLU())
model:add(nn.Dropout(0.1)) -- used it was 0.5

model:add(nn.Linear(layerSize,layerSize))
model:add(nn.ReLU())
model:add(nn.Dropout(0.1)) -- used it was 0.5

model:add(nn.Linear(layerSize,layerSize))
model:add(nn.ReLU())
model:add(nn.Dropout(0.1)) -- used it was 0.5

model:add(nn.Linear(layerSize, noutputs))

print('model is ready')
print('==> printing model...')
print(model)
wait(10)

end
