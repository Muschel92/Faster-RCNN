
--require 'torch'   -- torch
require 'loadcaffe'
require 'torch'
require 'nn'
require 'cudnn'
require 'cunn'
require 'cutorch'
require 'image'

inn = require 'inn'

dofile('FastRCNN/fast_rcnn_config.lua')

conf = config()
---------------------------------------------------------------------------------------------
-- load trained caffe model
old_model = loadcaffe.load('/data/ethierer/ObjectDetection/FasterRCNN/Model/OriginalModels/Caffe/VGG16_deploy.prototxt', '/data/ethierer/ObjectDetection/FasterRCNN/Model/OriginalModels/Caffe/VGG_ILSVRC_16_layers.caffemodel', 'cudnn')

print(old_model)
model_feat = nn.Sequential()
--model:cuda()

for i=1,29 do 
  mdl = old_model:get(i)
  model_feat:add(mdl)
end


roi_pooling = inn.ROIPooling(7,7):setSpatialScale(1/16)

par = nn.ParallelTable()
par:add(model_feat)
par:add(nn.Identity())

model = nn.Sequential()

model:add(par)

model:add(roi_pooling)
model:add(nn.View(-1, 25088))

lin1 = nn.Linear(25088,4096)
--lin1.weight:normal(0,0.01)
--lin1.bias:fill(0.0)

classifier = nn.Sequential()
classifier:add(lin1)
classifier:add(cudnn.ReLU())
classifier:add(nn.Dropout(0.5))

lin2 = nn.Linear(4096, 4096)
--lin2.weight:normal(0, 0.01)
--lin2.bias:fill(0.0)

classifier:add(lin2)
classifier:add(cudnn.ReLU())
classifier:add(nn.Dropout(0.5))

classifier = require('weight-init')(classifier, 'kaiming')

cc = nn.ConcatTable()
lin3 = nn.Linear(4096, 21)
--lin3.weight:normal(0, 0.01)
--lin3.bias:fill(0)

cc:add(lin3)
lin4 = nn.Linear(4096,84)
--lin4.weight:normal(0, 0.001)
--lin4.bias:fill(0)
cc:add(lin4)

model:add(classifier)
model:add(cc)

model = model:cuda()

print(model)
--feature_model = model:clone()

torch.save (conf.network_path .. 'VGG16_Fast_Rcnn_' .. conf.rois_per_batch_f .. '_kaiming.t7', model)
