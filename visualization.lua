
require 'torch'   -- torch
require 'image'   -- for image transforms
require 'nn'      -- provides all sorts of trainable modules/layers
require 'cunn'

----------------------------------------------------------------------
--model = torch.load('bestResultOnTest/pretrained/model-direct-notAugmnted-dataset5-mse0.15.net')
model = torch.load('bestResultOnTest/model.net')
residual = false

----------------------------------------------------------------------
print '==> the model:'
print(model)

----------------------------------------------------------------------
-- Visualization is quite easy, using itorch.image().
if itorch then

  if residual then
    itorch.image(model:get(2):get(1):get(1).weight)
  else
    print '==> visualizing ConvNet filters'
    print('Layer 2 filters:')
    itorch.image(model:get(2).weight)
    print('Layer 4 filters:')
    itorch.image(model:get(4).weight)
    print('Layer 6 filters:')
    itorch.image(model:get(6).weight)
    print('Layer 8 filters:')
    itorch.image(model:get(8).weight)
    print('Layer 10 filters:')
    itorch.image(model:get(10).weight)
  end
else
  print '==> To visualize filters, start the script in itorch notebook'
end
