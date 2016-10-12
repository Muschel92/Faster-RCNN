
--require 'torch'   -- torch
require 'loadcaffe'
require 'torch'
require 'nn'
require 'cudnn'
require 'cunn'
require 'cutorch'
require 'image'
require 'inn'



---------------------------------------------------------------------------------------------
-- load trained caffe model
model = torch.load('/data/ethierer/ObjectDetection/FasterRCNN/Model/FastRcnnTrained_stage2/VGG1601_Tue11Oct_2340_ep16.t7')

model = model.modules[1].modules[1]

for i = 1,#model.modules do
  model.modules[i].updateGradInput = function() end
  model.modules[i].accGradParameters = function() end
end


-- if you want to save the feature_map then uncommend
--feature_model = model:clone()

--torch.save ('/home/ethierer/Hiwi/Projects/Code/Faster RCNN/Model/VGG16_Feature_Map.t7', feature_model)

-- anchors at each sliding window position
k = 9
--conf.total_anchors or 9

model_class = nn.Sequential()
-- sliding window that reduces dimensionality to k anchors
-- initialize new layers
model_class:add(cudnn.ReLU())
rpn = cudnn.SpatialConvolution(512, 512, 3, 3, 1,1, 1, 1):cuda()
rpn.weight:normal(0, 0.1)
rpn.bias:fill(0)

model_class:add(rpn)
model_class:add(cudnn.ReLU())

-- predicts object score for every anchor
outlayer1 = cudnn.SpatialConvolution(512, 3*k, 1, 1):cuda()
outlayer1.weight:normal(0, 0.01)
outlayer1.bias:fill(0)


outlayer2 = cudnn.SpatialConvolution(512, 4*k, 1 , 1):cuda()
outlayer2.weight:normal(0, 0.01)
outlayer2.bias:fill(0)

outlayer = nn.ConcatTable():cuda()
outlayer:add(outlayer1)
outlayer:add(outlayer2)

model_class:add(outlayer)

--model_class = require('weight-init')(model_class, 'xavier_caffe')

for i = 1, #model_class.modules do 
  model:add(model_class.modules[i])
end
  
print(model)

torch.save ( '/data/ethierer/ObjectDetection/FasterRCNN/Model/ProposalNetwork/stage_2/' .. 'VGG1601_Tue11Oct_2340_ep1601001.t7', model)
