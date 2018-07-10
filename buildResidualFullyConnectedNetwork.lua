
--------------
-- This function builds a residual fully connected network inspired by 
-- the paper: "Deep Residual Learning for Image Recognition"
-- 
-- There are two sequences of blocks - the first contains layers with the same
-- size of ninputs and the second sequences contains layers with the size of nNeurons.
-- In between there is a single layer that transforms the data from nInputs space to 
-- nNeurons space. The last layer simply projects the second blocks sequnce output
-- to nOutputs size
-- 
-- The model that gave the best results on my dataset was built 
-- by calling: buildFullyConnectedResidualNetwork(2, 30, 200,43, 35)   
--
-- Author: Itzik Malkiel
-- Date: 25/1/16
--------------
function buildFullyConnectedResidualNetwork(nFirstBlocks, nSecondBlocks, nNeurons,nOutputs, nInputs)

  local model = nn.Sequential()
  model:add(nn.Reshape(nInputs))

  --build first n bloacks
  for i=1,nFirstBlocks do
    addResidualFullyConnectedBlock(model, nInputs)
  end

  --change to bigger layers
  model:add(nn.Linear(nInputs,nNeurons))
  model:add(nn.ReLU())
  
  --build second n blocks
  for i=1,nSecondBlocks do
    addResidualFullyConnectedBlock(model, nNeurons)
  end
  
  model:add(nn.Linear(nNeurons, nOutputs))
  
  print('==> Model is ready')
  print('==> Printing model...')
  print(model)
  
  return model
end


function addResidualFullyConnectedBlock(model, nNeurons)
   --craete a layer
    linearLayer = nn.Sequential()
    linearLayer:add(nn.Linear(nNeurons,nNeurons))
    linearLayer:add(nn.ReLU(true))
    --model:add(nn.Dropout(0.01)) --enabling this results with prediction of 0 values. don't use dropout here
  
    --add a shortcut edge 
    concat = nn.ConcatTable()
    concat:add(linearLayer)
    concat:add(nn.Identity())
    model:add(concat)
    
    --sum identity and linearLayer output
    model:add(nn.CAddTable())
    
end