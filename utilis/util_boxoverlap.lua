require('torch')

-- return the intersection in a torch.Tensor(number of boxes, 3)
function boxoverlap (boxes1, boxes2)
  overlap = torch.Tensor(boxes1:size(1), boxes2:size(1))
  
  local x1_1 = boxes1:index(2, torch.LongTensor{1})
  local x2_1 = boxes1:index(2, torch.LongTensor{3})
  local y1_1 = boxes1:index(2, torch.LongTensor{2})
  local y2_1 = boxes1:index(2, torch.LongTensor{4})
  
  local x1 = torch.Tensor(x1_1:size())
  local x2 = torch.Tensor(x1_1:size())
  local y1 = torch.Tensor(x1_1:size())
  local y2 = torch.Tensor(x1_1:size())
  
  local width = torch.Tensor(x1_1:size())
  local height = torch.Tensor(x1_1:size())
  
  for i = 1, boxes2:size(1) do
      
    local x1_2 = torch.Tensor(x1_1:size(1)):fill(boxes2[i][1])    
    local x2_2 = torch.Tensor(x2_1:size(1)):fill(boxes2[i][3])
    local y1_2 = torch.Tensor(y1_1:size(1)):fill(boxes2[i][2])
    local y2_2 = torch.Tensor(y2_1:size(1)):fill(boxes2[i][4])
    
    
    x1:cmax(x1_1, x1_2)
    x2:cmin(x2_1, x2_2)
    y1:cmax(y1_1, y1_2)
    y2:cmin(y2_1, y2_2)
    
    height = x2 - x1 + 1
    width = y2 - y1 + 1
    
    height:cmax(0)
    width:cmax(0)
    
    -- intersection
    height:cmul(width)    
    
    local area1 = torch.cmul((x2_1 - x1_1 + 1),(y2_1 - y1_1 + 1))
    local area2 = torch.cmul((x2_2 - x1_2 + 1),(y2_2 - y1_2 + 1))
    
    height:cdiv(area1 + area2 - height)

    overlap:indexCopy(2, torch.LongTensor{i}, height)
    
  end

  return overlap
  
end
