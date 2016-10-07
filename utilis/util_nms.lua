-- calculate maximum supression
-- supresses boxes with Iou > overlap of boxes with highest score
-- copied from andreaskoepfer
--[[
function nms(boxes, overlap, scores)
  
  pick = torch.LongTensor()
  
  x1 = boxes[{{},1}]
  x2 = boxes[{{},3}]
  y1 = boxes[{{},2}]
  y2 = boxes[{{},4}]
  
  area = torch.cmul(x2- x1 +1, y2 - y1 + 1)
  
  
  v, I = scores:sort(1)
  
  count = 1
  
  xx1 = boxes.new()
  xx2 = boxes.new()
  yy1 = boxes.new()
  yy2 = boxes.new()
  
  w = boxes.new()
  h = boxes.new()
  
  pick:resize(boxes:size(1))
  
  while I:numel() > 0 do
      last = I:size(1)
       i = I[last]
      
      
      pick[count] = i
      count = count + 1
      
      if last == 1 then
        break
      end
      
      I = I[{{1, last-1}}]
      
      xx1:index(x1, 1, I)
      xx2:index(x2, 1, I)
      yy1:index(y1, 1, I)
      yy2:index(y2, 1, I)
      
      xx1:cmax(x1[i])
      xx2:cmin(x2[i])
      yy1:cmax(y1[i])
      yy2:cmin(y2[i])
      
      w:resize(xx2:size())
      h:resize(yy2:size())
      
      torch.add(w, xx2, -1, xx1):add(1):cmax(0)
      torch.add(h, yy2, -1, yy1):add(1):cmax(0)
      
      w:cmul(h)
            
      xx1:index(area, 1, I)
      torch.cdiv(h, w, xx1 + area[i] - w)
      
      I = I[h:le(overlap)]
  end
  
  pick = pick[{{1, count-1}}]
    
  return pick  
end
]]--

function nms(boxes, overlap, scores)
  
  local pick = torch.LongTensor()
    
  local v, I = scores:sort(1)
  
  local count = 1
  
  local target = boxes.new()
  local remain = boxes.new()
  
  pick:resize(boxes:size(1))
  
  while I:numel() > 0 do
      local last = I:size(1)
      local i = I[last]
      
      
      pick[count] = i
      count = count + 1
      
      if last == 1 then
        break
      end
      
      I = I[{{1, last-1}}]
      
      target:index(boxes,1, torch.LongTensor{i})
      target:resize(1,4)
      remain:index(boxes,1, I)
      
      local IoU = boxoverlap(remain, target)
      
      I = I[IoU:le(overlap)]
  end
  
  pick = pick[{{1, count-1}}]
    
  return pick  
end
