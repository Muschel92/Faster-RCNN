

function test_single_class(path, gt, cls)
  
    assert(io.open(path))
    local file = io.open(path, 'r')
    -- check if file is open
    if file then
      local count = 0
      local l_table = {}
      for line in file:lines() do
        --print(line)
        table.insert(l_table, line)
        count = count + 1
      end
      
      local img_id = torch.Tensor(count)
      local confidence = torch.Tensor(count)
      local ex_boxes = torch.Tensor(count, 4)
      
      count = 0
      -- extract results from file
      for i, line in ipairs(l_table) do
        count = count+1
        local table = string.split(line, '%s')
        img_id[count] = tonumber(table[1])
        confidence[count] = tonumber(table[2])
        ex_boxes[{count,1}] = tonumber(table [3])
        ex_boxes[{count,2}] = tonumber(table [4])
        ex_boxes[{count,3}] = tonumber(table [5])
        ex_boxes[{count,4}] = tonumber(table [6])
      end
    
      local npos = 0
      local gt_found = {}
      
      for i = 1,#gt do
          local cls_idx = gt[i].labels:eq(cls)
          --cls_idx:cmul(gt[i].diff:eq(0):float())
          
          local num_cls = torch.sum(cls_idx)
          local temp = torch.zeros(num_cls)
          table.insert(gt_found, temp)
          
          npos = npos + num_cls    
      end
           
      -- sort confidence by drecreasing confidence
      local sorted_confidence, sorted_idx = torch.sort(confidence, 1, true)
      sorted_confidence:resize(sorted_confidence:numel())
      
      img_id = img_id:index(1, sorted_idx)
      img_id:resize(img_id:numel())
      ex_boxes = ex_boxes:index(1, sorted_idx)
      
      local true_positives = torch.zeros(img_id:numel())
      local false_positives = torch.zeros(img_id:numel())
      
      for i = 1, sorted_confidence:numel() do
        local cls_inds = gt[img_id[i]].labels:eq(cls)
        local num_boxes = torch.sum(cls_inds)
        
        if num_boxes > 0 then
          local cls_idx = binaryToIdx(cls_inds)
          local gt_boxes = gt[img_id[i]].gt_boxes:index(1, cls_idx)
          
          local overlap = boxoverlap(gt_boxes, ex_boxes[i]:reshape(1,4))
          local max, max_idx = torch.max(overlap, 1)
          max:resize(max:numel())
          max_idx:resize(max_idx:numel())
          
          if max[1] > 0.1 then 
              -- ignore difficult objects
              if gt[img_id[i]].diff[max_idx[1]] == 0 then
                if gt_found[img_id[i]][max_idx[1]] == 0 then
                  gt_found[img_id[i]][max_idx[1]] = 1
                  true_positives[i] = 1
                else
                  false_positives[i] = 1  -- multiple detection of same object
                end
              end
          else
                false_positives[i] = 1  -- false positive
          end
  
        else
            false_positives[i] = 1 -- false positives
        end  
      end
      
      -- compute precision and recall
      false_positives = torch.cumsum(false_positives)
      true_positives = torch.cumsum(true_positives)
      local recall = torch.div(true_positives, npos)
      local precision = torch.cdiv(true_positives, false_positives + true_positives)
      
      -- compute average precision
      local range = torch.range(0,1, 0.1)
      local average_presicion = 0
      
      for i = 1,range:numel() do
        local thresh = precision[recall:ge(range[i])]
        local p = 0
        if thresh:numel() > 0 then
          p =  torch.max(thresh)
        end
        
        average_presicion = average_presicion + p/11
      end
      
      return average_presicion, recall, precision, npos
      
  else
  error('Can not read file')
  end
end
