
--require 'torch'   -- torch
require 'loadcaffe'
require 'torch'
require 'nn'
require 'cudnn'
require 'cunn'
require 'cutorch'
require 'image'

inn = require 'inn'

--dofile('FastRCNN/fast_rcnn_config.lua')

--conf = config()
---------------------------------------------------------------------------------------------
-- load trained caffe model
old_model = torch.load('/data/ethierer/ObjectDetection/FasterRCNN/Model/OriginalModels/resnet-50.t7')

model_feat = nn.Sequential()
--model:cuda()

for i=1,7 do 
  mdl = old_model:get(i)
  model_feat:add(mdl)
end


roi_pooling = inn.ROIPooling(7,7):setSpatialScale(1/16)

par = nn.ParallelTable()
par:add(model_feat)
par:add(nn.Identity())

--im = torch.ByteTensor(2,3,600,800):random(256):cuda()
--boxes = torch.CudaTensor({{1,1,1,16,16},{1,2,8,12,24},{2,10, 9, 20, 24}})

--input = {}
--table.insert(input, im)
--table.insert(input, boxes)


model = nn.Sequential()

model:add(par)

model:add(roi_pooling)

classifier = nn.Sequential()
model:add(old_model:get(8))
model:add (nn.SpatialAveragePooling(4,4))

model:add(nn.View(-1, 2048))
cc = nn.ConcatTable()
lin3 = nn.Linear(2048, 21)
lin3.weight:normal(0, 0.01)
lin3.bias:fill(0)

cc:add(lin3)
lin4 = nn.Linear(2048,84)
lin4.weight:normal(0, 0.001)
lin4.bias:fill(0)
cc:add(lin4)


model:add(cc)
model = model:cuda()

print(model)
--feature_model = model:clone()

torch.save ('/data/ethierer/ObjectDetection/FasterRCNN/Model/FastRCNN/Res50_Fast_Rcnn_128_2.t7', model)
