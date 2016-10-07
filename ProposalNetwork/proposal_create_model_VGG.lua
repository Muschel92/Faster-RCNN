
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
old_model = torch.load('/data/ethierer/ObjectDetection/FasterRCNN/Model/OriginalModels/vgg16.t7')

model = nn.Sequential()
--model:cuda()

for i=1,29 do 
  mdl = old_model:get(i)
  model:add(mdl)
end

-- if you want to save the feature_map then uncommend
--feature_model = model:clone()

--torch.save ('/home/ethierer/Hiwi/Projects/Code/Faster RCNN/Model/VGG16_Feature_Map.t7', feature_model)

-- anchors at each sliding window position
k = 9
--conf.total_anchors or 9

-- sliding window that reduces dimensionality to k anchors
-- initialize new lay--model:add(cudnn.ReLU())
model:add(cudnn.ReLU())
rpn = cudnn.SpatialConvolution(512, 512, 3, 3, 1,1, 1, 1):cuda()
rpn.weight:normal(0, 0.01)
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

torch.save ( '/data/ethierer/ObjectDetection/FasterRCNN/Model/ProposalNetwork/stage_1/VGG16_RPN_Network.t7', model)
  --conf.network_path .. 'VGG16_Proposal_net_' .. k ..'.t7', model)
