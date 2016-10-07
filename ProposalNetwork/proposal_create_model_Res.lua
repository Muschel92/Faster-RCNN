
--require 'torch'   -- torch
require 'loadcaffe'
require 'torch'
require 'nn'
require 'cudnn'
require 'cunn'
require 'cutorch'
require 'image'



---------------------------------------------------------------------------------------------
-- load trained caffe model
old_model = torch.load('/data/ethierer/ObjectDetection/FasterRCNN/Model/resnet-50.t7')

old_model:remove()
old_model:remove()
old_model:remove()
old_model:remove()
  
feat_model = old_model:clone()
torch.save('/data/ethierer/ObjectDetection/FasterRCNN/Model/Res_50_Feature_Map.t7', feat_model)

-- anchors at each sliding window position
k = 9
--conf.total_anchors or 9

-- sliding window that reduces dimensionality to k anchors
-- initialize new lay--model:add(cudnn.ReLU())
rpn = cudnn.SpatialConvolution(1024, 512, 3, 3, 1,1, 1, 1):cuda()
rpn.weight:normal(0, 0.01)
rpn.bias:fill(0)

old_model:add(rpn)
old_model:add(cudnn.ReLU())

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


old_model:add(outlayer)

print(old_model)

torch.save ( '/data/ethierer/ObjectDetection/FasterRCNN/Model/' .. 'Res_50_Proposal_net_' .. k ..'_zeroBias.t7', old_model)
  --conf.network_path .. 'VGG16_Proposal_net_' .. k ..'.t7', model)
