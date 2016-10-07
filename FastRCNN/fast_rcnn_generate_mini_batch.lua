local rois_blob = torch.CudaTensor()
local rois_feat_blob = torch.CudaTensor()
local labels_blob = torch.CudaTensor()
local bbox_targets_blob = torch.CudaTensor()
local bbox_loss_blob = torch.CudaTensor()
local gt_idx_blob = torch.CudaTensor()
local im_blob = torch.CudaTensor()
local im_scales = torch.CudaTensor()
local targets = torch.CudaTensor()


function fast_rcnn_generate_minibatch(roidb)
    
  local num_images = #roidb
  
  local num_classes = conf.numClasses

    -- generate scale for every image
  local random_scale_inds = torch.Tensor(num_images):random(conf.scales:size(1))
  local random_flipped = torch.Tensor(num_images):random(2)
  
  local rois_per_image = conf.rois_per_batch_f /num_images
  local fg_rois_per_image = math.floor(rois_per_image * conf.fg_fraction_f + 0.5)
  
  local blob, scales = fast_rcnn_get_image_blob(conf, roidb, random_scale_inds, random_flipped)

  im_blob:resize(blob:size()):copy(blob)
  im_scales:resize(scales:size()):copy(scales)  
	
  rois_feat_blob:resize(num_images * rois_per_image, 5):fill(0)

  rois_blob:resize(num_images * rois_per_image, 4):fill(0)
  labels_blob:resize(num_images* rois_per_image):fill(22)
  gt_idx_blob:resize(num_images * rois_per_image):fill(0)
  targets:resize(num_images*rois_per_image, 4):fill(0)
  bbox_targets_blob:resize(num_images * rois_per_image,  4 *(conf.numClasses + 1)):fill(0)
  bbox_loss_blob:resize(bbox_targets_blob:size()):fill(0)
  
  local im_scales = {}
  
  local gt_boxes = torch.Tensor()
  
  local index = 1
  local nr_gt = 0
  
  for i = 1,num_images do
    -- get the scale for the gt boxes
    local scale = prep_im_for_blob_size(roidb[i].size, conf.scales[random_scale_inds[i]], conf.max_size)
    
    table.insert(im_scales, scale)
    local labels, im_rois, bbox_targets, bbox_loss, gt_indx = fast_rcnn_sample_rois(conf, roidb[i], fg_rois_per_image, rois_per_image, random_scale_inds[i], scale, random_flipped[i])
    
    if im_rois:dim() == 2 then
      
            
      local feat_rois =  map_im_rois_to_feat_map(roidb[i], im_rois, random_scale_inds[i], scale)
      local num_rois = feat_rois:size(1)
      --print(rois_blob[{{index, index + num_rois -1}, {}}])
      

      rois_blob[{{index, index + num_rois -1}, {}}] = im_rois

      
      rois_feat_blob[{{index, index + num_rois-1}, 1}] = i
      rois_feat_blob[{{index, index + num_rois-1}, {2,5}}] = feat_rois
      
      labels_blob[{{index,index + num_rois-1}}] = labels
      gt_idx_blob[{{index,index + num_rois-1}}] = gt_indx + nr_gt

      bbox_targets_blob[{{index, index + num_rois-1}, {}}] = bbox_targets
      
      bbox_loss_blob[{{index, index + num_rois-1}, {}}] = bbox_loss
      
      if (gt_boxes:dim() == 0) then
        gt_boxes = scale_rois(roidb[i].gt_boxes:float(), roidb[i].size:float(), scale)
        
        if random_flipped[i] == 2 then
          gt_boxes = flip_rois(gt_boxes, scale)
        end
      else
        local temp_gt = scale_rois(roidb[i].gt_boxes:float(), roidb[i].size:float(), scale)
        
        if random_flipped[i] == 2 then
          temp_gt = flip_rois(temp_gt, scale)
        end
        gt_boxes = gt_boxes:cat(temp_gt, 1)      
      end

      index = index + num_rois
      nr_gt = nr_gt * roidb[i].gt_boxes:size(1)
    end

    -- if there are not enought fore- and background rois fill them up with background
    if i == num_images and index < i * rois_per_image then
      rois_feat_blob:resize(index -1, 5)
      rois_blob:resize(index -1, 4)
      gt_idx_blob:resize(index -1, 1)
      labels_blob:resize(index -1, 1)
      bbox_loss_blob:resize(index -1, 4 *(conf.numClasses + 1) )
      bbox_targets_blob:resize(index -1, 4 *(conf.numClasses + 1) )
    end
    
  end

    local batch = {}
    -- 1) image data
    table.insert(batch, im_blob:clone())
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
    --9) the actual scale of the images
    table.insert(batch, im_scales)   
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
      local im = image.load(images[i].path, 3, 'byte')
      
      if flipped[i] == 2 then
        im = image.hflip(im)
      end
      
      local target_size = conf.scales[random_scale_inds[i]]
      
      
      local im_scale = 0
      im, im_scale= prep_im_for_blob(im, conf.image_means, target_size, conf.max_size);
      
      local size = torch.Tensor{im:size(2), im:size(3)}
      im_scales[i][{}] = torch.cdiv(im_scale, size)
      
      if conf.different_image_size then
        im = im:index(1, torch.LongTensor{3,2,1})
        
      end

      table.insert(processed_images, im)
  end
  
    im_blob = im_list_to_blob(processed_images)
  
  return im_blob, im_scales
end

local ex_boxes = torch.Tensor()
local gt_boxes = torch.Tensor()

function fast_rcnn_sample_rois(conf, roidb, fg_rois_per_image, rois_per_image, scale_idx, scale, flipped)
  
  -- calculate overlap
  ex_boxes = ex_boxes:resize(roidb.ex_boxes:size()):copy(roidb.ex_boxes)
  gt_boxes = gt_boxes:resize(roidb.gt_boxes:size()):copy(roidb.gt_boxes)
    
  local nr_of_images = ex_boxes:size(1)
  
  if roidb.ex_boxes:size(1) > conf.topN_proposals then
    ex_boxes = ex_boxes[{{1, conf.topN_proposals},{}}]
  end
  
  if flipped == 2 then
    ex_boxes = flip_rois(ex_boxes, roidb.size)
    gt_boxes = flip_rois(gt_boxes, roidb.size)
  end

  local overlaps = boxoverlap(ex_boxes:float(), gt_boxes:float())
  
  local max_o, max_l = torch.max(overlaps, 2)
  
  local idx_fg = torch.ge(max_o, conf.fg_thresh_f)
  local nr_fg_inds = torch.sum(idx_fg)
  local fg_inds = torch.zeros(idx_fg:size())
  
  fg_rois_per_image = math.min(nr_fg_inds, fg_rois_per_image)
  -- get random fg rois
  if (fg_rois_per_image > 0 ) then
      local perm = torch.randperm(nr_fg_inds):long()
      perm = perm[{{1, fg_rois_per_image}}]
      local idx = torch.zeros(nr_fg_inds)
      idx:indexFill(1, perm, 1)
      fg_inds[idx_fg:eq(1)] = idx
  end
    
  local idx_bg = torch.cmul(torch.lt(max_o, conf.bg_thresh_hi_f), torch.ge(max_o, conf.bg_thresh_lo_f))
  local bg_rois_per_image = rois_per_image -fg_rois_per_image
  local nr_bg_inds = torch.sum(idx_bg)
  local bg_inds = torch.zeros(idx_bg:size())
  nr_of_images = nr_of_images - fg_rois_per_image
  
  nr_bg_inds = math.min(nr_of_images, nr_bg_inds)
  
  bg_rois_per_image = math.min(nr_bg_inds, bg_rois_per_image)
  
  if bg_rois_per_image > 0 then
    local perm = torch.randperm(nr_bg_inds):long()
    perm = perm[{{1, bg_rois_per_image}}]
    local idx = torch.zeros(nr_bg_inds)
    idx:indexFill(1, perm, 1)
    bg_inds[idx_bg:eq(1)] = idx
  end

  -- combine fg and bg indexes
  local keep_inds = torch.add(fg_inds:float(), bg_inds:float())
  keep_inds= keep_inds:gt(0)
  
  --set labels of unused rois to zero
  max_l[torch.ne(keep_inds, 1)] = -1
  -- set labels of bg_rois to 0
  max_l[torch.eq(bg_inds,1)] = 0
  
  -- get all gt_labels of bg and fg inds
  local labels = torch.Tensor(ex_boxes:size(1))
  -- set background label to 21
  labels[max_l:eq(0)] = 21
  
  -- set for all foreground rois the label
  if torch.sum(fg_inds) > 0 then
    labels[fg_inds:gt(0)] = roidb.labels:index(1, max_l[fg_inds:gt(0)]):float()
  end

  -- only keep labels for fore- and background rois
  labels = labels[torch.gt(keep_inds, 0)]
  
  -- only keep the groundtruth boxes for fore- and background rois
  max_l = max_l[torch.gt(keep_inds, 0)]
  
  keep_inds = keep_inds:double()
  
  keep_inds = keep_inds:cat(keep_inds, 2):cat(keep_inds, 2):cat(keep_inds, 2)
  
  -- get fore- and background rois
  if torch.sum(keep_inds) > 0 then
    ex_boxes = ex_boxes[torch.ge(keep_inds, 1)]
    ex_boxes:resize(ex_boxes:size(1) / 4, 4)
    
    keep_inds = keep_inds:cat(keep_inds[{{},1}], 2)
    
    -- get bounding box targets
    local targets = roidb.bbox_targets[scale_idx][flipped]
    if targets:size(1) > conf.topN_proposals then
      targets = targets[{{1,conf.topN_proposals},{}}]
    end
    
    targets = targets[torch.gt(keep_inds,0)]
    
    if targets:dim() > 0 then
      targets = targets:reshape(targets:size(1)/5, 5)
      targets[{{},1}] = labels
    end
      
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
	  
	  local inds = torch.gt(clss, 0)
	  inds = inds:cat(inds,1):cat(inds,1):cat(inds,1)
    
	  for i = 1, bbox_targets:size(1) do
	    if clss[i] < 21  then
	      local cls = clss[i]
	    
	      local targets_inds = torch.range(1+(cls-1)*4,cls*4):long()
        local idx = torch.zeros(bbox_targets:size(2))
        idx:indexFill(1, targets_inds:long(), 1)
	      bbox_targets[i][idx:eq(1)] = bbox_target_data[{i, {2,5}}]:float()
        bbox_loss_weight[i][idx:eq(1)] = 1
        --bbox_targets[i]:index(1, targets_inds):copy(bbox_target_data[{i, {2,5}}])
	      --bbox_loss_weight[i]:index(1, targets_inds):fill(1)
	    end
    
	  end  
    
    return bbox_targets, bbox_loss_weight
  else
	  local bbox_targets = torch.zeros(rois_per_image, 4 *(num_classes + 1))
	  local bbox_loss_weight = torch.zeros(bbox_targets:size())
    bbox_loss_weight[{{},{1,4}}] = 1
    return bbox_targets, bbox_loss_weight
  end
end

