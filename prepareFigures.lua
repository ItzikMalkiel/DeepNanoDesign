dofile 'utils.lua'
require 'mattorch'
--new_mat_Au_epsDiele_1.2_geom_6_lmbd_600_25_1650_L0_215_L1_215_h_40_w_40_theta_70_polarization_x1y0_P_700.mat_y_1-1-0-0-201-1-210-71
--rawResult_new_mat_Au_epsDiele_1.2_geom_10_lmbd_600_25_1650_L0_215_L1_195_h_40_w_40_theta_10_polarization_x1y0_P_700.mat_x_1-0-1-1-203-1-222-11')
--new_mat_Au_epsDiele_1.2_geom_6_lmbd_600_25_1650_L0_215_L1_215_h_40_w_40_theta_29_polarization_x1y0_P_700.mat_x_1-1-0-0-210-1-217-26
x  = mattorch.load('figuresAndRawResults/rawResult_new_mat_Au_epsDiele_1.2_geom_11_lmbd_600_25_1650_L0_173.4_L1_147.7_h_40_w_40_theta_62.2_polarization_x1y0_P_700.mat_y_1-1-1-1-158-1-185-59')
require 'gnuplot'
print (x)
print (x.deepLearning[1][1])
pred = x.deepLearning:clone()
comsol = x.comsol:clone()
for i = 1,43 do
    if pred[1][i] > 1 then
      pred[1][i] = 1
    end
end

for i = 1,43 do
    if comsol[1][i] > 1 then
      comsol[1][i] = 1
    end
end
  
  
plotting = true
plotOnePolarization(pred, "x polarization", true,pred  ,false)
plotOnePolarization(comsol, "x polarization", true,pred  ,false)


