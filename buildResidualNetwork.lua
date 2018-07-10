
function buildFullyConnectedResidualNetwork(nBlocks, nNeurons,noutputs, ninputs)

  model = nn.Sequential()


  model:add(nn.Reshape(ninputs))
--  model:add(nn.Linear(ninputs,nNeurons))
--  model:add(nn.ReLU())
--  model:add(nn.Dropout(0.6))


--loop for N layers
  for i=1,2 do
    --craete a layer
    unit = nn.Sequential()
    unit:add(nn.Linear(ninputs,ninputs))
    unit:add(nn.ReLU(true))
    --model:add(nn.Dropout(0.01)) --enabling this results with prediction of 0 values. don't use dropout here
  
    --wrap layer with shortcut edge 
    cat = nn.ConcatTable()
    cat:add(unit)
    cat:add(nn.Identity())
    model:add(cat)
    
    --sum identity and layers output
    model:add(nn.CAddTable())
    
  end

  model:add(nn.Linear(ninputs,nNeurons))
  model:add(nn.ReLU())
  
  --loop for N layers
  for i=1,nBlocks do
    --craete a layer
    unit = nn.Sequential()
    unit:add(nn.Linear(nNeurons,nNeurons))
    unit:add(nn.ReLU(true))
    --model:add(nn.Dropout(0.01)) --enabling this results with prediction of 0 values. don't use dropout here
  
    --wrap layer with shortcut edge 
    cat = nn.ConcatTable()
    cat:add(unit)
    cat:add(nn.Identity())
    model:add(cat)
    
    --sum identity and layers output
    model:add(nn.CAddTable())
    
    --add batch normalization
    --model:add(nn.BatchNormalization(nNeurons))
    
    --add relu
    --model:add(nn.ReLU(true)) --maybe enable this relu
  end
  
  model:add(nn.Linear(nNeurons, noutputs))
  
  print('model is ready')
  print('==> printing model...')
  print(model)
  
  wait(5)


  return model
end
--    
--model = nn.Sequential()
--
--model:add(nn.Reshape(50))
--
--local net = nn.Linear(50,200)(model)
--net = nn.BatchNormalization(200, 1e-3)(net)
--net = nn.ReLU()(net)
--skip = model
--net = nn.CAddTable(){net, skip}
--net = nn.BatchNormalization(200, 1e-3)(net)


function buildResidualFullyConnecteBlock(input,  nChannels, nOutChannels, stride)
   --[[
   Residual layers! Implements option (A) from Section 3.3. The input
   is passed through two 3x3 convolution filters. In parallel, if the
   number of input and output channels differ or if the stride is not
   1, then the input is downsampled or zero-padded to have the correct
   size and number of channels. Finally, the two versions of the input
   are added together.
               Input
                 |
         ,-------+-----.
   Downsampling      3x3 convolution+dimensionality reduction
        |               |
        v               v
   Zero-padding      3x3 convolution
        |               |
        `-----( Add )---'
                 |
              Output
   --]]
   nOutChannels = nOutChannels or nChannels
   stride = stride or 1
   -- Path 1: Convolution
   -- The first layer does the downsampling and the striding
   local net = cudnn.SpatialConvolution(nChannels, nOutChannels,
                                           3,3, stride,stride, 1,1)(input)
   net = nn.SpatialBatchNormalization(nOutChannels)(net)
   net = cudnn.ReLU(true)(net)
   net = cudnn.SpatialConvolution(nOutChannels, nOutChannels,
                                      3,3, 1,1, 1,1)(net)

   -- Path 2: Identity / skip connection
   local skip = input
   if stride > 1 then
       -- optional downsampling
       skip = nn.SpatialAveragePooling(1, 1, stride,stride)(skip)
   end
   if nOutChannels > nChannels then
       -- optional padding
       skip = nn.Padding(1, (nOutChannels - nChannels), 3)(skip)
   end

   -- Add them together
   net = nn.CAddTable(){net, skip}
   net = nn.SpatialBatchNormalization(nOutChannels)(net)
   --net = cudnn.ReLU(true)(net)
   -- ^ don't put a ReLU here! see http://gitxiv.com/comments/7rffyqcPLirEEsmpX

   return net
end