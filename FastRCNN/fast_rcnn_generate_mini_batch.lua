
function fast_rcnn_generate_minibatch(roidb)
    
  local num_images = #roidb
  
  local num_classes = conf.numClasses

    -- generate scale for every image
  local random_scale_inds = torch.Tensor(num_images):random(conf.scales:size(1))
  local random_flipped = torch.Tensor(num_images):random(2)
  
  local rois_per_image = conf.rois_per_batch_f /num_images
  local fg_rois_per_image = math.floor(rois_per_image * conf.fg_fraction_f + 0.5)
  
  local blob, _ = fast_rcnn_get_image_blob(conf, roidb, random_scale_inds, random_flipped) 
	
  local rois_feat_blob = torch.Tensor()
  local rois_blob = torch.Tensor()
  local labels_blob = torch.Tensor()
  local gt_idx_blob = torch.Tensor()
  local bbox_targets_blob = torch.Tensor()
  local bbox_loss_blob = torch.Tensor()
  
  local im_sizes = {}
  
  local gt_boxes = torch.Tensor()
  
  local nr_gt = 0
  
  for i = 1,num_images do
    -- get the scale for the gt boxes
    local scale = prep_im_for_blob_size(roidb[i].size, conf.scales[random_scale_inds[i]], conf.max_size)
    
    table.insert(im_sizes, scale:clone())
    local labels, im_rois, bbox_targets, bbox_loss, gt_indx = fast_rcnn_sample_rois(conf, roidb[i], fg_rois_per_image, rois_per_image, random_scale_inds[i], scale, random_flipped[i])
            
    if im_rois:dim() == 2 then
      
      local feat_rois =  im_rois:index(2, torch.LongTensor{2,1,4,3})
      local num_rois = feat_rois:size(1)
      --print(rois_blob[{{index, index + num_rois -1}, {}}])
      
      if i == 1 or rois_blob:dim() == 0 then
        rois_blob = im_rois:clone()
        
        local ones = torch.Tensor(num_rois):fill(i)
        rois_feat_blob = ones:cat(feat_rois,2)
      
        labels_blob= labels
        gt_idx_blob= gt_indx + nr_gt

        bbox_targets_blob = bbox_targets
  
        bbox_loss_blob = bbox_loss
      
      
        gt_boxes = scale_rois(roidb[i].gt_boxes, roidb[i].size, scale)
        
        if random_flipped[i] == 2 then
          gt_boxes = flip_rois(gt_boxes, scale)
        end
      else
        rois_blob = rois_blob:cat(im_rois, 1)
        local ones = torch.Tensor(num_rois):fill(i)
        local temp1  = ones:cat(feat_rois,2)
        rois_feat_blob = rois_feat_blob:cat(temp1, 1)
        
        labels_blob= labels_blob:cat(labels, 1)
        
        gt_indx[gt_indx:gt(0)] = gt_indx[gt_indx:gt(0)] + nr_gt
        gt_idx_blob= gt_idx_blob:cat(gt_indx, 1)

        bbox_targets_blob = bbox_targets_blob:cat(bbox_targets,1)
  
        bbox_loss_blob = bbox_loss_blob:cat(bbox_loss, 1)
      
      
        local temp2  = scale_rois(roidb[i].gt_boxes, roidb[i].size, scale)
        
        if random_flipped[i] == 2 then
          temp2 = flip_rois(temp2, scale)
        end    
        
        gt_boxes = gt_boxes:cat(temp2, 1)
      end

      nr_gt = nr_gt + roidb[i].gt_boxes:size(1)
    end
    collectgarbage()
  end

    local batch = {}
    -- 1) image data
    table.insert(batch, blob:clone())
    -- 2) rois on feature map
    table.insert(batch, rois_feat_blob:clone())
    -- 3) rois on image
    table.insert(batch, rois_blob:clone())
    -- 4) labels of images (21 for background)
    table.insert(batch, labels_blob:clone())
    -- 5) regression (rois x 21 x 4)
    table.insert(batch, bbox_targets_blob:clone())
    -- 6) loss for regression ( 1 for class of box, rest 0)
    table.insert(batch, bbox_loss_blob:clone())
    -- 7) gt box of image for positive rois
    table.insert(batch, gt_idx_blob:clone())
    -- 8) the scaled gt boxes
    table.insert(batch, gt_boxes:clone())
    --9) the actual size of the images
    table.insert(batch, im_sizes)   
    -- 10) if image is flipped(2) or not (1)
    table.insert(batch, random_flipped:clone())
    return batch
end


-- loads images and stores them in one tensor
function fast_rcnn_get_image_blob(conf, images, random_scale_inds, flipped)
  
  local num_images = #images
  processed_images = {}
  
  local im_scales = torch.Tensor(num_images, 2):fill(0/0)
  
  for i = 1, num_images do
      local im = image.load(images[i].path, 3, 'byte'):float()
      
      if flipped[i] == 2 then
        im = image.hflip(im)
      end
      
      local target_size = conf.scales[random_scale_inds[i]]

      local ims, im_scale= prep_im_for_blob(im, conf.image_means, target_size, conf.max_size);
      
      local size = torch.Tensor{im:size(2), im:size(3)}
      im_scales[i][{}] = torch.cdiv(im_scale, size)
      
      if conf.different_image_size then
        ims = ims:index(1, torch.LongTensor{3,2,1})
        
      end

      table.insert(processed_images, ims)
  end
  
    im_blob = im_list_to_blob(processed_images)
  
  return im_blob, im_scales
end

function fast_rcnn_sample_rois(conf, roidb, fg_rois_per_image, rois_per_image, scale_idx, scale, flipped)
  
  -- calculate overlap
  local ex_boxes = roidb.ex_boxes:clone()
  local gt_boxes = roidb.gt_boxes:clone()
    
  local nr_of_images = ex_boxes:size(1)
  
  if roidb.ex_boxes:size(1) > conf.topN_proposals then
    local range = torch.range(1, conf.topN_proposals):long()
    ex_boxes = ex_boxes:index(1, range)
  end
  
  if flipped == 2 then
    ex_boxes = flip_rois(ex_boxes, roidb.size)
    gt_boxes = flip_rois(gt_boxes, roidb.size)
  end

  local overlaps = boxoverlap(ex_boxes, gt_boxes)
  
  local max_o, max_l = torch.max(overlaps, 2)
  
  local fg_inds = torch.ge(max_o, conf.fg_thresh_f)
  local nr_fg_inds = torch.sum(fg_inds)
  local idx_fg = binaryToIdx(fg_inds)
  
  fg_rois_per_image = math.min(nr_fg_inds, fg_rois_per_image)
  
  
  -- get random fg rois
  if (fg_rois_per_image > 0 ) then
      local perm = torch.randperm(nr_fg_inds):long()
      idx_fg = idx_fg:index(1, perm[{{1, fg_rois_per_image}}])
  end
    
  local bg_inds = torch.cmul(max_o:lt(conf.bg_thresh_hi_f), max_o:ge(conf.bg_thresh_lo_f))
  local bg_rois_per_image = rois_per_image -fg_rois_per_image
  local nr_bg_inds = torch.sum(bg_inds)
  local idx_bg = binaryToIdx(bg_inds)
  
  bg_rois_per_image = math.min(nr_bg_inds, bg_rois_per_image)
  
  if bg_rois_per_image > 0 then
    local perm = torch.randperm(nr_bg_inds):long()
    idx_bg = idx_bg:index(1, perm[{{1, bg_rois_per_image}}])
  end

  -- combine fg and bg indexes
  
  local idx_keep = torch.Tensor()
  if nr_fg_inds > 1 and nr_bg_inds > 0 then
    --print(nr_fg_inds)
    --print(idx_fg)
    --print(idx_fg:cat(idx_bg))
    idx_keep = idx_fg:cat(idx_bg, 1)
  elseif nr_fg_inds == 1 and nr_bg_inds > 0 then
    idx_keep = idx_fg:cat(idx_bg)
  elseif nr_bg_inds > 0 then
    idx_keep = idx_bg
  else
    return torch.Tensor(), torch.Tensor()
  end
  
  
  -- set labels of bg_rois to 0
  max_l:indexFill(1, idx_bg, 0)
  -- get all gt_labels of bg and fg inds
  local labels = max_l:index(1, idx_keep):float()
  
  -- set for all foreground rois the label
  if nr_fg_inds > 0 then
    --print(idx_fg)
    --print (max_l:index(1, idx_fg))
    local temp = max_l:index(1, idx_fg)
    temp = temp:reshape(temp:size(1)):long()
    if roidb.labels:size(1) > roidb.labels:size(2)  then
      local temp2 = roidb.labels:index(1, temp)
      labels[labels:gt(0)]  = temp2[{{},1}]
    else
      local temp2 = roidb.labels:index(2, temp)
      labels[labels:gt(0)]  = temp2[1]
    end
    
  end
  
  -- set background label to 21
  labels[labels:eq(0)] = 21
  
  -- only keep the groundtruth boxes for fore- and background rois
  max_l = max_l:index(1, idx_keep)
  -- get fore- and background rois
  if nr_bg_inds + nr_fg_inds > 0 then
    ex_boxes = ex_boxes:index(1, idx_keep)
    
    -- get bounding box targets
    local targets = roidb.bbox_targets[scale_idx][flipped]:clone()
    if targets:size(1) > conf.topN_proposals then
      targets = targets[{{1,conf.topN_proposals},{}}]
    end
        
    targets = targets:index(1, idx_keep)
    targets[{{},1}] = labels
      
    ex_boxes = scale_rois(ex_boxes, roidb.size:float(), scale)

    local bbox_targets, bbox_loss_weights = fast_rcnn_get_bbox_regression_labels(conf, targets, conf.numClasses, rois_per_image)
    
    return labels, ex_boxes, bbox_targets, bbox_loss_weights, max_l  
  end
  
  return labels, ex_boxes.new()
end

-- Calculate the regression labels
-- only non zero for regression to positive label
-- num classes are number of gt rois of image
function fast_rcnn_get_bbox_regression_labels(conf, bbox_target_data, num_classes, rois_per_image)
  
  if(bbox_target_data:dim() == 2) then
	  local clss = bbox_target_data[{{},1}]
	  local bbox_targets = torch.zeros(clss:size(1), 4 *(num_classes + 1))
	  local bbox_loss_weight = torch.zeros(bbox_targets:size())
	  
	  local inds = torch.lt(clss, 21)
    local idx = binaryToIdx(inds)
	  local targets_temp = bbox_targets:index(1, idx)
    local loss_temp = bbox_loss_weight:index(1, idx)
    local clss_temp = clss:index(1, idx)
    
	  for i = 1, targets_temp:size(1) do
	      local cls = clss_temp[i]
	    
	      local targets_inds = torch.range(1+(cls-1)*4,cls*4):long()
        local idx = torch.zeros(bbox_targets:size(2))
        idx:indexFill(1, targets_inds:long(), 1)
	      targets_temp[i][idx:eq(1)] = bbox_target_data[{i, {2,5}}]
        loss_temp[i][idx:eq(1)] = 1    
	  end  
    
    bbox_targets:indexCopy(1, idx, targets_temp)
    bbox_loss_weight:indexCopy(1, idx, loss_temp)
    
    return bbox_targets, bbox_loss_weight
  else
	  local bbox_targets = torch.zeros(rois_per_image, 4 *(num_classes + 1))
	  local bbox_loss_weight = torch.zeros(bbox_targets:size())
    bbox_loss_weight[{{},{1,4}}] = 1
    return bbox_targets, bbox_loss_weight
  end
end

