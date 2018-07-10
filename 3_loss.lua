----------------------------------------------------------------------
-- This script defines two different
-- loss functions:
--   + mean-square error
--   + l1 error
--
----------------------------------------------------------------------


print '==> define loss'
if opt.loss == 'mse' then
   criterion = nn.MSECriterion()
   --criterion.sizeAverage = false
elseif opt.loss == 'abs' then
  criterion = nn.AbsCriterion()
  criterion.sizeAverage = true
else
   error('unknown -loss')
end

----------------------------------------------------------------------
print '==> here is the loss function:'
print(criterion)
