
-- Wie werde ich das Bild und die ROIs übergeben? Dann Funktion anpassen!
-- ímdbs: table with information about image(path, id, size, name)
-- roidbs: table with 
function proposal_prepare_image_roidb(conf, imdbs, bbox_means, bbox_stds)

  if (bbox_means == nil) then
    bbox_means = nil
    bbox_stds = nil
  end


  image_roidb =imdbs

  image_roidb, bbox_means, bbox_stds = append_bbox_regression_targets(conf, image_roidb, bbox_means, bbox_stds)


  return image_roidb, bbox_means, bbox_stds

end


function append_bbox_regression_targets(conf, image_roidb, means, stds)

  local roidb = image_roidb
  local nr_images = #image_roidb    
  -- for every image  
  for i = 1,nr_images do
    roidb[i].size = roidb[i].size:float()
    roidb[i].boxes= roidb[i].boxes:float()
    roidb[i].labels= roidb[i].labels:float()
    local anchors, im_scales = proposal_locate_anchors(conf, roidb[i].size, roidb[i].feature_map_size)
    local gt_rois = roidb[i].boxes
    local gt_labels = roidb[i].class
    local im_size = roidb[i].size
    local bbox_targets = {}
    for j = 1, #im_scales do
      scale = torch.cdiv(im_scales[j], roidb[i].size)
      local rois = scale_rois(gt_rois, im_size, im_scales[j])
      local targets = compute_targets(conf, rois, gt_labels, anchors[j], roidb[i], scale)
      
      rois = flip_rois(rois, torch.cmul(im_size, scale))
      local targets_flipped = compute_targets(conf, rois, gt_labels, anchors[j], roidb[i], scale)
      local flipped = {}
      table.insert(flipped, targets)
      table.insert(flipped, targets_flipped)
      table.insert( bbox_targets, flipped)
    end
    roidb[i].bbox_targets = bbox_targets 
  end

  -- calculate new means if null
  if (means == nil and stds == nil ) then
    local class_counts = 0
    local sums = torch.Tensor(1,4):fill(0)
    local squared_sum = torch.Tensor(1,4):fill(0)

    for i = 1,nr_images do
      for j = 1,conf.scales:size(1) do
        local targets = roidb[i].bbox_targets[j][1]   
        --local targets_flipped = roidbs[i].bbox_targets[j][2]   

        for k = 1,targets:size(1) do
          if (targets[{k,1}] > 0 and targets[{k,1}] < 21 ) then
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
    for i = 1,nr_images do
      for j = 1,conf.scales:size(1) do
        local targets = roidb[i].bbox_targets[j][1]   
        --local targets_flipped = roidbs[i].bbox_targets[j][2]   

        for k = 1,targets:size(1) do
          if (targets[{k,1}] > 0 and targets[{k,1}] < 21 ) then
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
    
    print(stds)
    
    local std = torch.zeros(2, stds:size(2))
    std[{2,{}}] = stds
    stds = std
    --print(stds)
  end

  -- normalize all bbox targets
  for i = 1, nr_images do
    for j = 1,conf.scales:size(1) do
      local targets = roidb[i].bbox_targets[j][1]
      local targets_flipped = roidb[i].bbox_targets[j][2]  
      for k = 1,targets:size(1) do
        if targets[{k,1}] > 0 and targets[{k,1}] < 21 then
          roidb[i].bbox_targets[j][1][k][{{2,5}}]:csub(means[2])
          roidb[i].bbox_targets[j][1][k][{{2,5}}]:cdiv(stds[2])

        end 
        if targets_flipped[{k,1}] > 0 and targets_flipped[{k,1}] < 21 then
          roidb[i].bbox_targets[j][2][k][{{2,5}}]:csub(means[2])
          roidb[i].bbox_targets[j][2][k][{{2,5}}]:cdiv(stds[2])
        end
      end    
    end
  end
  
  local mean_temp = torch.zeros(4)
  local std_temp = torch.zeros(4)
  local count = 0
  for i = 1,nr_images do 
      local targets = roidb[i].bbox_targets[1][1] 
      
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
  print(mean_temp)
  mean_temp:div(count)
  print(mean_temp)
  
  count = 0
  
  for i = 1,nr_images do 
      local targets = roidb[i].bbox_targets[1][1] 
      
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
  print(std_temp)
  std_temp:div(count-1)
  std_temp:sqrt()
  print(std_temp)  

  return roidb, means, stds
end



-- Arguments:
-- conf: configuration object
-- gt_rois: scaled ground truth rois as tensors
-- gt_labels: ground_truth labels
-- ex_rois: rois of the anchors
-- image_roidb: information about the image
-- im_scale: scale of the image
-- returns for every fg and bg bounding box the label and regression
-- negative ex_rois have label -1
-- rest is zero
function compute_targets(conf, gt_rois, gt_labels, ex_rois, roidb, im_scale)
  

  local image_roidb = roidb
  local bbox_targets = torch.Tensor()
  -- check if there are any ground truth bounding boxes
  if (gt_rois:dim() == 0) then
    bbox_targets:resize(ex_rois:size(1), 5)
    bbox_targets:indexFill(1, torch.LongTensor{2},-1)
    return bbox_targets
  end

  -- calc overlap between ex_rois and gt_rois
  local ex_gt_overlap = boxoverlap(ex_rois, gt_rois)

  -- drop anchors which run out off image boundaries if nececarry
  -- by setting overlap to zero
  local contained_in_image = torch.Tensor()
  if (conf.drop_boxes_runoff_image) then
    local im_size_l = image_roidb.size:clone()
    contained_in_image = is_contained_in_image(ex_rois, im_size_l:cmul(im_scale):round())
    for i = 1, gt_rois:size(1) do  
      ex_gt_overlap[{{}, i}]:cmul(contained_in_image)
    end
  end

  -- for each ex_rois(anchor) get its max overlap with gt_rois and the index of the gt_rois
  local ex_max_overlaps, ex_assignment = torch.max(ex_gt_overlap, 2)
  -- find the best overlaps for every gt_rois
  local gt_max_overlaps, gt_ind = torch.max(ex_gt_overlap, 1)

  -- get all best ex_rois for every gt_rois
  local positives_max_idx = torch.zeros(ex_gt_overlap:size(1))
  
  for i = 1, gt_max_overlaps:size(2) do
    
    local idx = torch.eq(ex_gt_overlap[{{}, i}], gt_max_overlaps[{1,i}])
    positives_max_idx:add(idx:float())
  end
  
  positives_max_idx = positives_max_idx:gt(0)
  
  -- indices of ex_rois for which we try to make predictions
  -- negative examples IoU < 0.3
  local negatives = ex_max_overlaps:lt(conf.bg_thresh_hi):float()
  -- all anchors with iou greater/equal threshhold
  local positives_gt = ex_max_overlaps:ge(conf.fg_thresh)
  
  -- drop anchors which run out off image boundaries if nececarry    
  if (conf.drop_boxes_runoff_image) then
    negatives:cmul(contained_in_image)
  end

  -- get the ones best matching got gt
  local positives = torch.add(positives_gt, positives_max_idx)
  positives = positives:gt(0)
  
  local positives_labels = ex_assignment[positives:eq(1)]
  
  -- extract all positive ex_rois
  local src = torch.Tensor(torch.sum(positives), 4)
  for i = 1,4 do
    src[{{},i}]:copy(ex_rois[{{},i}][positives:eq(1)])
  end
  
  -- extract all the target gt_rois for all positive ex_rois
  local targets = torch.Tensor(torch.sum(positives), 4)
  targets:copy(gt_rois:index(1, ex_assignment[positives:eq(1)]))
  
  --positives = positives:cat(positives):cat(positives):cat(positives)

  -- calculate the regression labels
  local regression_labels = bbox_transform(src, targets)
  
  -- returns for every fg and bg bounding box the label and regression
  -- negative ex_rois have label -1
  -- rest is zero
  bbox_targets:resize(ex_rois:size(1), 5):fill(0)
  for i = 1,4 do
    bbox_targets[{{},i+1}][torch.eq(positives,1)]=regression_labels[{{},i}]
  end
  
  local temp = bbox_targets[{{},1}][torch.eq(positives,1)]
  temp[{}] = positives_labels[{}]
  bbox_targets[{{},1}][torch.eq(positives,1)] = temp
  
  -- check that boxes are either negative or positive
  local neg_not_pos = torch.ones(positives:size(1))
  neg_not_pos[torch.eq(positives,1)] = 0
  negatives = torch.cmul(negatives, neg_not_pos)
  
  bbox_targets[{{},1}][torch.eq(negatives,1)] = -1
  

  return (bbox_targets)
end

-- checks if the bbox is contained in image
function is_contained_in_image(boxes, im_size)

  if (boxes:dim() >= 1) then
    local x1 = torch.le(boxes[{{},1}], im_size[1])
    local x2 = torch.le(boxes[{{},2}], im_size[2])
    local x3 = torch.le(boxes[{{},3}], im_size[1])
    local x4 = torch.le(boxes[{{},4}], im_size[2])

    x1 = torch.cmul(x1:float(),x2:float())
    x3 = torch.cmul(x3:float(),x4:float())

    local contained = torch.cmul(x1, x3)
    return contained
  end     
end