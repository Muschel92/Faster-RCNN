
function fast_rcnn_prepare_image_roidb(conf, roidbs, bbox_means, bbox_stds)

  if (bbox_means == nil) then
    bbox_means = nil
    bbox_stds = nil
  end
  
  local image_roidbs, bbox_means, bbox_stds = fast_rcnn_append_bbox_regression_targets(conf, roidbs, bbox_means, bbox_stds)
  
  
  return image_roidbs, bbox_means, bbox_stds
end

-- calls function to compute the bbox targets
-- normalizes bbox targets
function fast_rcnn_append_bbox_regression_targets( conf, roidbs, means, stds)
  
  local num_images = #roidbs
  
  local num_classes = roidbs[1].numClasses
  
  local valid_imgs = torch.ones(num_images)
  
  -- calculate bbox_targets for all images
  for i = 1,num_images do
    local bbox_targets = {}
    for j = 1,conf.scales:size(1) do
        local scale = prep_im_for_blob_size(roidbs[i].size, conf.scales[j], conf.max_size)
        local rois = scale_rois(roidbs[i].ex_boxes, roidbs[i].size:float(), scale)
        local gt_rois = scale_rois(roidbs[i].gt_boxes, roidbs[i].size:float(), scale)
        local targets, val = fast_rcnn_compute_targets(conf, roidbs[i], rois, 
          gt_rois)     
        if torch.sum(targets:ne(targets)) > 0 then
          print(i)
        end
        
        -- targets for flipped rois
        rois = flip_rois(rois, scale)
        gt_rois = flip_rois(gt_rois, scale)
        local targets_flipped, val_flipped = fast_rcnn_compute_targets(conf, roidbs[i], rois, 
          gt_rois)
        
        valid_imgs[i] = val
        local all_targets = {}
        table.insert(all_targets, targets)
        table.insert(all_targets, targets_flipped)
        table.insert(bbox_targets, all_targets)
    end
    roidbs[i].bbox_targets = bbox_targets
  end
  
  print('Finished')
  -- remove all invalid images
  if (torch.sum(valid_imgs) ~= num_images) then
      local index = 1
      for i = 1,num_images do
        if(valid_imgs[i] == 0) then
          table.remove(roidbs, index)
          index = index -1
        end
        index = index +1
      end
  end
  
  -- calculate mean and stds of all bbox targets
  if (means == nil and stds == nil ) then
    local class_counts = 0
    local sums = torch.Tensor(1,4):fill(0)
    local squared_sum = torch.Tensor(1,4):fill(0)

    for i = 1,num_images do
      for j = 1,conf.scales:size(1) do
        local targets = roidbs[i].bbox_targets[j][1]   
        --local targets_flipped = roidbs[i].bbox_targets[j][2]   

        for k = 1,targets:size(1) do
          if (targets[{k,1}] > 0 and targets[{k,1}] < conf.numClasses +1 ) then
            class_counts = class_counts + 1
            local temp = targets[{k,{2,5}}]
            sums:add(temp)
            --temp:pow(2)
            --squared_sum:add(temp)
          end 
          --[[
          if (targets_flipped[{k,1}] > 0 and targets_flipped[{k,1}] < conf.numClasses +1 ) then
            local temp = targets_flipped[{k,{2,5}}]
            class_counts = class_counts + 1
            sums:add(temp)
            temp:pow(2)
            squared_sum:add(temp)
          en]]--
        end              
      end
    end
    print('Mean')
    
    --print(sums)
    --print(class_counts)
    means = torch.div(sums, class_counts)
    --print(means)
    means = means[{1, {1,4}}]    
    
    local mean = torch.zeros(2, means:size(1))
    mean[{2,{}}] = means
    means = mean

    print(means)
    
    class_counts = 0
    for i = 1,num_images do
      for j = 1,conf.scales:size(1) do
        local targets = roidbs[i].bbox_targets[j][1]   
        --local targets_flipped = roidbs[i].bbox_targets[j][2]   

        for k = 1,targets:size(1) do
          if (targets[{k,1}] > 0 and targets[{k,1}] < conf.numClasses +1 ) then
            local temp = targets[k][{{2,5}}]:reshape(4)
            class_counts = class_counts + 1
            temp:csub(means[{2,{}}])
            temp:pow(2)
            squared_sum:add(temp)
          end 
          --[[
          if (targets_flipped[{k,1}] > 0 and targets_flipped[{k,1}] < conf.numClasses +1 ) then
            local temp = targets_flipped[{k,{2,5}}]
            class_counts = class_counts + 1
            sums:add(temp)
            temp:pow(2)
            squared_sum:add(temp)
          en]]--
        end              
      end
    end
    -- sums / classes
    print('STD')
    --print(class_counts)
    --print(squared_sum)
    
    --print(mean)
    -- sqrt((squared_sum / class_counts)- means²)
    
    stds = squared_sum / (class_counts -1)
    stds:sqrt()
    --stds = torch.pow(torch.add(torch.div(squared_sum, class_counts),- torch.pow(means[{2,{}}], 2)), 0.5)
    
    --print(stds)
    
    local std = torch.zeros(2, stds:size(2))
    std[{2,{}}] = stds
    stds = std
    print(stds)
  end 
  
  -- normalize all bbox targets
  for i = 1, num_images do
    for j = 1,conf.scales:size(1) do
      local targets = roidbs[i].bbox_targets[j][1]
      local targets_flipped = roidbs[i].bbox_targets[j][2]  
      for k = 1,targets:size(1) do
        if targets[{k,1}] > 0 and targets[{k,1}] < conf.numClasses +1 then
          roidbs[i].bbox_targets[j][1][k][{{2,5}}]:csub(means[2])
          roidbs[i].bbox_targets[j][1][k][{{2,5}}]:cdiv(stds[2])

        end 
        if targets_flipped[{k,1}] > 0 and targets_flipped[{k,1}] < conf.numClasses +1 then
          roidbs[i].bbox_targets[j][2][k][{{2,5}}]:csub(means[2])
          roidbs[i].bbox_targets[j][2][k][{{2,5}}]:cdiv(stds[2])
        end
      end    
    end
  end
  
  local mean_temp = torch.zeros(4)
  local std_temp = torch.zeros(4)
  local count = 0
  for i = 1,num_images do 
      local targets = roidbs[i].bbox_targets[1][1] 
      
      for j = 1,targets:size(1) do
        if targets[j][1] > 0 and targets[j][1] < 21 then
          local temp = targets[j][{{2,5}}]:reshape(4)
          mean_temp:add(temp)
          count = count + 1
        end
      end
      
      --print(mean_temp)
      --targets = roidbs[i].bbox_targets[1][2] 
      --[[
      for j= 1,targets:size(1) do
        if targets[j][1] > 0 and targets[j][1] < 21 then
          local temp = targets[j][{{2,5}}]:reshape(4)
          mean_temp:add(temp)
          count = count + 1
        end
      end
      ]]--
      
  end
  
  print('New_mean')
  mean_temp:div(count)
  print(mean_temp)
  
  count = 0
  
  for i = 1,num_images do 
      local targets = roidbs[i].bbox_targets[1][1] 
      
      for j = 1,targets:size(1) do
        if targets[j][1] > 0 and targets[j][1] < 21 then
          local temp = targets[j][{{2,5}}]:reshape(4)
          temp:csub(mean_temp)
          temp:pow(2)
          std_temp:add(temp)
          count = count + 1
        end
      end
      
      --targets = roidbs[i].bbox_targets[1][2] 
      
      --[[
      for j= 1,targets:size(1) do
        if targets[j][1] > 0 and targets[j][1] < 21 then
          local temp = targets[j][{{2,5}}]:reshape(4)
          temp:csub(mean_temp)
          temp:pow(2)
          std_temp:add(temp)
          count = count + 1
        end
      end
      ]]--
      
  end
  print('New_std')
  std_temp:div(count-1)
  std_temp:sqrt()
  print(std_temp)  
      
  
  
  return roidbs, means, stds
end


-- computes for gt_rois, target_rois and a bounding box regression
function fast_rcnn_compute_targets(conf, roidb, rois, gt_rois)
  
  local overlap = boxoverlap(rois, gt_rois)
  
  -- get gt rois with greates overlap
  local max_overlap, max_gt = torch.max(overlap, 2)
  
  local bbox_targets = torch.zeros(rois:size(1), 5)
  bbox_targets[{{},1}] = 21

    -- find rois with overlap > bbox_threshhold
  local ex_inds = torch.ge(max_overlap, conf.fg_thresh_f)
  
  max_gt = max_gt[ex_inds:eq(1)]
  
  ex_inds = ex_inds:cat(ex_inds, 2):cat(ex_inds):cat(ex_inds,2)
  rois = rois[ex_inds:eq(1)]

  -- calculate the regression labels for all rois
  
  if max_gt:dim() ~= 0 then
    
    rois:resize(rois:size(1)/4,4)
    local gt = gt_rois:index(1, max_gt)
    
    reg_labels = bbox_transform(rois, gt)
    
    if torch.sum(rois:ne(rois)) > 0 then
          print('rois')
    end
  
    -- label for every bounding box
    bbox_targets[{{},1}][torch.eq(ex_inds[{{},1}],1)] = roidb.labels:index(1, max_gt:long()):float()
    bbox_targets[{{},{2,5}}][torch.eq(ex_inds,1)] = reg_labels:float()
  end
  
  -- Select foreground ROIS as those with fg_thresh > fg_thresh overlap
  is_fg = torch.ge(max_overlap, conf.fg_thresh_f)
  is_bg = torch.cmul(torch.lt(max_overlap, conf.bg_thresh_hi_f), torch.ge(max_overlap, conf.bg_thresh_lo_f))
  
  -- check if rois are bg and fg rois
  local is_valid = 1
  if (torch.sum(torch.cmul(is_fg, is_bg)) ~= 0) then
    is_valid = 0
  end
  
  return bbox_targets, is_valid
end