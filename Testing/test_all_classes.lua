require 'torch'
require 'nn'
require 'paths'
require 'image'
require 'optim'

-----------------------------------------------------------------------------
dofile('utilis/util_boxoverlap.lua')
dofile('utilis/util_binaryToIdx.lua')
dofile('utilis/util_pascal_voc_2007_labels.lua')
dofile('Testing/test_single_class.lua')


local num_classes = 20
local test_path = '/data/ethierer/ObjectDetection/FasterRCNN/Results/fastrcnn_1/Res50_TueOkt11_2332_ep16_32.t7/WedOct1209:21:192016/'
local ground_truth = torch.load('/data/ethierer/ObjectDetection/FasterRCNN/Data/TestData/trainVal.t7')


local averagePrecision = torch.Tensor(num_classes)
local recall = torch.Tensor(num_classes)
local precision = torch.Tensor(num_classes)
local nr_of_objects = torch.Tensor(num_classes)

for i = 1,num_classes do
  local name = loadLabelFromNumber(i)
  local file_name = name .. '.txt'
  local path = test_path .. file_name
  local aP, R, P, npos = test_single_class(path, ground_truth, i)
  averagePrecision[i] = aP
  recall[i] = R[R:numel()]
  precision[i] = P[P:numel()]
  nr_of_objects[i] = npos
end

print('Average Precision: ')
print(averagePrecision)

print('Recall: ')
print(recall)

print('Precision: ')
print(precision)

print('Nr of Objects: ')
print(nr_of_objects)

