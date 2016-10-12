
--require 'torch'   -- torch
require 'torch'
require 'nn'
require 'cudnn'
require 'cunn'
require 'cutorch'
require 'image'
require 'inn'



---------------------------------------------------------------------------------------------
-- load trained caffe model
old_model = torch.load('/data/ethierer/ObjectDetection/FasterRCNN/Model/FastRcnnTrained_stage2/Res50_TueOkt11_2332_ep16_32.t7')

model = old_model.modules[1].modules[1]:clone()

print(old_model)

--[[
conv_nodes = old_model:findModules('cudnn.SpatialConvolution')
batch_nodes = old_model:findModules('nn.SpatialBatchNormalization')
relu_nodes = old_model:findModules('cudnn.ReLU')
identity_nodes = old_model:findModules('nn.Identity')
concat_nodes = old_model:findModules('nn.ConcatTable')
sequ_nodes = old_model:findModules('nn.Sequential')
add_nodes = old_model:findModules('nn.CAddTable')


for i = 1,#conv_nodes do
  conv_nodes[i].updateGradInput = function() end
  conv_nodes[i].accGradParameters = function() end
end

for i = 1,#batch_nodes do
  batch_nodes[i].backward = function() end
  batch_nodes[i].updateGradInput = function() end
  batch_nodes[i].accGradParameters = function() end
end

for i = 1,#relu_nodes do
  relu_nodes[i].updateGradInput = function() end
  relu_nodes[i].accGradParameters = function() end
end

for i = 1,#identity_nodes do
  identity_nodes[i].updateGradInput = function() end
  identity_nodes[i].accGradParameters = function() end
end

for i = 1,#concat_nodes do
  concat_nodes[i].updateGradInput = function() end
  concat_nodes[i].accGradParameters = function() end
end

for i = 1,#sequ_nodes do
  sequ_nodes[i].updateGradInput = function() end
  sequ_nodes[i].accGradParameters = function() end
end

for i = 1,#add_nodes do
  add_nodes[i].updateGradInput = function() end
  add_nodes[i].accGradParameters = function() end
end
]]--
-- if you want to save the feature_map then uncommend
--feature_model = model:clone()

--torch.save ('/home/ethierer/Hiwi/Projects/Code/Faster RCNN/Model/VGG16_Feature_Map.t7', feature_model)

-- anchors at each sliding window position
k = 9
--conf.total_anchors or 9

rpn = cudnn.SpatialConvolution(1024, 512, 3, 3, 1,1, 1, 1):cuda()
rpn.weight:normal(0, 0.1)
rpn.bias:fill(0)

model:add(rpn)
model:add(cudnn.ReLU())

-- predicts object score for every anchor
outlayer1 = cudnn.SpatialConvolution(512, 3*k, 1, 1):cuda()
outlayer1.weight:normal(0, 0.1)
outlayer1.bias:fill(0)



outlayer2 = cudnn.SpatialConvolution(512, 4*k, 1 , 1):cuda()
outlayer2.weight:normal(0, 0.1)
outlayer2.bias:fill(0)

outlayer = nn.ConcatTable():cuda()
outlayer:add(outlayer1)
outlayer:add(outlayer2)


model:add(outlayer)

print(model)

torch.save ( '/data/ethierer/ObjectDetection/FasterRCNN/Model/ProposalNetwork/stage_2/' .. 'Res50_TueOkt11_2332_ep16_32_01001.t7', model)
