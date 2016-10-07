
--[[ calculate the ground truth boxes from the regression label and the anchor
  position ]]--
  
eps = 1e-5

function bbox_from_regression(ex_boxes, regression, means, stds)
  
  
  local reg = torch.CudaTensor(regression:size(1), regression:size(2))
  
  for i = 1,4 do
    reg[{{},i}] = regression[{{},i}] * stds[{1,i}] + means[i]
  end
  
  local ex_width = ex_boxes[{{}, 3}] - ex_boxes[{{}, 1}] + 1
  local ex_height = ex_boxes[{{}, 4}] - ex_boxes[{{}, 2}] + 1
  local ex_ctr_x  = ex_boxes[{{},1}] + 0.5 * (ex_width - 1) 
  local ex_ctr_y  = ex_boxes[{{},2}] + 0.5 * (ex_height -1) 
  
  local gt_x_ctr = torch.cmul(reg[{{},{1}}], ex_width) + ex_ctr_x
  --print(gt_x_ctr)
  local gt_y_ctr = torch.cmul(reg[{{},{2}}], ex_height) + ex_ctr_y
  local gt_width = torch.cmul(torch.exp(reg[{{}, 3}]), ex_width)
  local gt_height = torch.cmul(torch.exp(reg[{{}, 4}]), ex_height)

  local gt_x1 = gt_x_ctr - 0.5 * gt_width
  local gt_x2 = gt_x_ctr + 0.5 * gt_width
  local gt_y1 = gt_y_ctr - 0.5 * gt_height
  local gt_y2 = gt_y_ctr + 0.5 * gt_height
  
  local bboxes = torch.cat(gt_x1, gt_y1, 2):cat(gt_x2, 2):cat(gt_y2, 2)
  
  return bboxes
end


--[[
function bbox_from_regression(ex_boxes, regression, means, stds)
  
  
  local reg = torch.CudaTensor(regression:size(1), regression:size(2))
  
  for i = 1,4 do
    reg[{{},i}] = torch.add(torch.mul(regression[{{},i}], stds[{1,i}]), means[i])
  end
  
  local ex_width = torch.add(torch.csub(ex_boxes[{{}, 3}], ex_boxes[{{}, 1}]), 1)
  --print(ex_width)
  local ex_height = torch.add(torch.csub(ex_boxes[{{}, 4}], ex_boxes[{{}, 2}]), 1)
  local ex_ctr_x  = torch.add(ex_boxes[{{},1}], torch.mul(ex_width - 1, 0.5))
  local ex_ctr_y  = torch.add(ex_boxes[{{},2}], torch.mul(ex_height - 1, 0.5))
  
  local gt_x_ctr = torch.add(torch.cmul(reg[{{},{1}}], ex_width), ex_ctr_x)
  --print(gt_x_ctr)
  local gt_y_ctr = torch.add(torch.cmul(reg[{{},{2}}], ex_height), ex_ctr_y)
  local gt_width = torch.cmul(torch.exp(reg[{{}, 3}]), ex_width)
  local gt_height = torch.cmul(torch.exp(reg[{{}, 4}]), ex_height)
  
  local gt_x1 = torch.add(gt_x_ctr, -torch.mul(gt_width, 0.5))
  local gt_x2 = torch.add(gt_x_ctr, torch.mul(gt_width, 0.5))
  local gt_y1 = torch.add(gt_y_ctr, -torch.mul(gt_height, 0.5))
  local gt_y2 = torch.add(gt_y_ctr, torch.mul(gt_height, 0.5))
  
  local bboxes = torch.cat(gt_x1, gt_y1):cat(gt_x2):cat(gt_y2)
  
  return bboxes
end ]]--
