-- Loads all the meta information of the images 
---------------------------------------------------------------------------
require 'torch'   -- torch
require 'image'   -- for color transforms
--require 'nn'      -- provides a normalization operator

--------------------------------------------------------------------------
dofile("Preprocessing/preparation_load_bboxes.lua")

local set_path = '/data/ethierer/data/Pascal Vor Challenge 2007/VOCdevkit/VOC2007/ImageSets/Main/'
local jpeg_path = '/data/ethierer/data/Test2007/VOCdevkit/VOC2007/JPEGImages/'
local xml_path = '/data/ethierer/data/Test2007/VOCdevkit/VOC2007/Annotations/'
local save_path = '/data/ethierer/ObjectDetection/FasterRCNN/Data/'

-- Load all train images
print("Load Train images")

test_boxes = {}
for file in paths.files(xml_path) do
  if file:find('.xml$') then
    local path = paths.concat(xml_path, file)
    table.insert(test_boxes, loadBboxes(path, jpeg_path))
  end
end

torch.save(save_path .. 'test_roidb.t7', val_bboxes)

