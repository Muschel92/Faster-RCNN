
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
old_model = torch.load('/data/ethierer/ObjectDetection/FasterRCNN/Model/model_stage3/Res50_Mo3Okt_ep16.t7')

old_model = old_model.modules[1].modules[1]

for i = 1,#old_model.modules do
  old_model.modules[i].updateGradInput = function() end
  old_model.modules[i].accGradParameters = function() end
end


-- if you want to save the feature_map then uncommend
--feature_model = model:clone()

--torch.save ('/home/ethierer/Hiwi/Projects/Code/Faster RCNN/Model/VGG16_Feature_Map.t7', feature_model)

-- anchors at each sliding window position
k = 9
--conf.total_anchors or 9

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

torch.save ( '/data/ethierer/ObjectDetection/FasterRCNN/Model/ProposalNetwork/stage_2/' .. 'Res50_Mo3Okt_ep16.t7', old_model)
