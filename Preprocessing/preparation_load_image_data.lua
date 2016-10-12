-- Loads all the meta information of the images 
---------------------------------------------------------------------------
require 'torch'   -- torch
require 'image'   -- for color transforms
--require 'nn'      -- provides a normalization operator

--------------------------------------------------------------------------
dofile("Preprocessing/preparation_load_bboxes.lua")

local set_path = '/data/ethierer/data/Pascal Vor Challenge 2007/VOCdevkit/VOC2007/ImageSets/Main/'
local jpeg_path = '/data/ethierer/data/Pascal Vor Challenge 2007/VOCdevkit/VOC2007/JPEGImages/'
local xml_path = '/data/ethierer/data/Pascal Vor Challenge 2007/VOCdevkit/VOC2007/Annotations/'
local save_path = '/data/ethierer/ObjectDetection/FasterRCNN/Data/'

-- Load all train images
print("Load Train images")
trainFile = assert(io.open(paths.concat(set_path, "train.txt")))
counter = 1
trainImages = {}

for line in trainFile:lines() do
  name = unpack(line:split(" "))
  trainImages[counter]= name .. '.xml'
  -- if counter == 5 then break end  --to load only small subset of images
  counter = counter + 1
end
trainFile:close()


train_bboxes = {}

for i, file in ipairs(trainImages) do
    local path = xml_path .. trainImages[i]
    table.insert(train_bboxes, loadBboxes(path, jpeg_path))
end

torch.save(save_path .. 'train_roidb.t7', train_bboxes)

-- Load all val images
print("Load Val Images")
valFile = assert(io.open(paths.concat(set_path, "val.txt")))
counter = 1
valImages = {}

for line in valFile:lines() do
  name = unpack(line:split(" "))
  valImages[counter]= name .. '.xml'
  -- if counter == 5 then break end  --to load only small subset of images
  counter = counter + 1
end
valFile:close()

val_bboxes = {}

for i, file in ipairs(valImages) do
    local path = xml_path .. valImages[i]
    table.insert(val_bboxes, loadBboxes(path, jpeg_path))
end

torch.save(save_path .. 'val_roidb.t7', val_bboxes)
  
--local image = torch.load('/home/ethierer/Hiwi/Projects/Data/Pascal Vor Challenge 2007/VOCdevkit/VOC2007/JPEGImages/000005.jpg')
