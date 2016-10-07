
function proposal_generate_mini_batch(conf, image_roidb)
    
    images = image_roidb
    local num_images = #image_roidb
    
    -- generate scale for every image
    local random_scale_inds = torch.Tensor(num_images):random(1, conf.scales:size(1))
    -- generate random flipped images
    local random_flipped = torch.Tensor(num_images):random(1,2)
    
    -- calculate how many rois per image are selected
    local rois_per_image = conf.rois_per_image
    
    -- calculate how many positive rois are selected
    local fg_rois_per_image = math.floor(rois_per_image * conf.fg_fraction + 0.5)
    
    -- images in [maxheight x maxwidth x 3 x images] 
    im_blob, im_scales = get_image_blob(conf, image_roidb, random_scale_inds, random_flipped)
    
    local labels_blob_list = {}
    local labels_weights_blob_list = {}
    local bbox_targets_blob_list = {}
    local bbox_loss_blob_list = {}
    
    -- for every image calculate the labels. their weights, bbox_targets and the loss
    for i = 1, num_images do
        -- get the random rois
        if i == 1 and false then
          print(image_roidb[i].bbox_targets[1]:size())
        end
        
        labels, labels_weight, bbox_targets, bbox_loss = sample_rois(conf, image_roidb[i], fg_rois_per_image, rois_per_image, im_scales[i], random_scale_inds[i], random_flipped[i])
        
        -- get fcn output size        
        --repl()
        im_size =torch.round(torch.cmul(image_roidb[i].size,im_scales[i]))
        
        -- number of different anchors
        k = conf.total_anchors
        
        -- size : (hight x width) x boxes)
        labels_blob = labels:reshape(1, image_roidb[i].feature_map_size[random_scale_inds[i]][1]* image_roidb[i].feature_map_size[random_scale_inds[i]][2], k)     
        
        labels_weight_blob = labels_weight:reshape(1, image_roidb[i].feature_map_size[random_scale_inds[i]][1] * image_roidb[i].feature_map_size[random_scale_inds[i]][2], k):cuda()
        
        -- size: (hight x widht x boxes), 4
        bbox_targets_blob = bbox_targets:reshape(image_roidb[i].feature_map_size[random_scale_inds[i]][1]* image_roidb[i].feature_map_size[random_scale_inds[i]][2]*k,4)
        
        bbox_loss_blob = bbox_loss:reshape(image_roidb[i].feature_map_size[random_scale_inds[i]][1]* image_roidb[i].feature_map_size[random_scale_inds[i]][2]*k,4)
                 
        table.insert(labels_blob_list, labels_blob:clone())
        table.insert(labels_weights_blob_list, labels_weight_blob:clone())
        table.insert(bbox_targets_blob_list, bbox_targets_blob:clone())
        table.insert(bbox_loss_blob_list, bbox_loss_blob:clone())
    end
    
    -- change images to BGR (not if we use different image sizes)
    --if not conf.different_image_size then
      --im_blob:index(1, torch.LongTensor{3,2,1})    
    --end
    
    local input_blob = {}
    table.insert(input_blob, im_blob)
    table.insert(input_blob, labels_blob_list)
    table.insert(input_blob, labels_weights_blob_list)
    table.insert(input_blob, bbox_targets_blob_list)
    table.insert(input_blob, bbox_loss_blob_list)
    table.insert(input_blob, random_scale_inds)
    table.insert(input_blob, random_flipped)
  
    return input_blob
end

-- loads the images according to a specific scale 
-- images have width(max_height x max_size x 3 x images]
-- images start at left upper corner
function get_image_blob(conf, images, random_scale_inds, random_flipped)
    
  local num_images = #images
  ims = images
  local processed_ims = {}
  
  
  -- fill with nan
  local im_scales = torch.Tensor(num_images, 2):fill(0/0)
  
  -- for every image
  for i = 1, num_images do
      -- load the image (channel x height x width)
      local im = image.load(images[i].path, 3, 'byte')
      
      if(random_flipped[i] == 2) then
        im = image.hflip(im)
      end

      local target_size = conf.scales[random_scale_inds[i]]
      
      -- scales the image
      im, im_scale = prep_im_for_blob(im, conf.image_means, target_size, conf.max_size)
      -- saves the scaling factor
      local size = torch.Tensor{im:size(2), im:size(3)}
      im_scales[i][{}] = torch.cdiv(im_scale, size)
      
      -- only if we use different image sizes
      im = im:index(1, torch.LongTensor{3,2,1})
      
      table.insert(processed_ims, im)
      
  end
  
  if(conf.different_image_size) then
    im_blob = processed_ims
  else
    im_blob = im_list_to_blob(processed_ims)
  end
  
  return im_blob, im_scales
  
end

-- calculates for one image a random sample of ROIs comprising foreground and background examples
-- returns:
-- bbox_regression for selected rois (rest is zero)
-- labels: labels for all fg rois( rest is zero)
-- label_weights: 1 for fg and conf.bg_weights for bg
-- bbox_loss_weights: 1 for fg rois zero for the rest
function sample_rois(conf, image_roidb, fg_rois_per_image, rois_per_image, im_scale, im_scale_idx, flipped)
  
  ims = image_roidb
  bbox_targets = image_roidb.bbox_targets[im_scale_idx][flipped]
  ex_assign_labels = bbox_targets[{{},1}]
  
  -- indices of all labels greater than 0 (positives)
  fg_inds = torch.gt(ex_assign_labels, 0)
  --fg_inds = fg_inds:cat(fg_inds, 2):cat(fg_inds,2):cat(fg_inds,2):cat(fg_inds,2)
  -- indices of all labels smaller than zero (negatives)
  bg_inds = torch.lt(ex_assign_labels, 0)
  --bg_inds = bg_inds:cat(bg_inds, 2):cat(bg_inds,2):cat(bg_inds,2):cat(bg_inds,2)

  -- number of positives samples
  local fg_num = 0
  local bg_num = 0
  if (torch.sum(bg_inds ) == 0) then
    fg_num = rois_per_image
  else
    fg_num = math.min(fg_rois_per_image, torch.sum(fg_inds:float()))
    bg_num = math.min(rois_per_image - fg_num, bg_inds:size(1))
  end
  
  fg_idx = torch.zeros(bbox_targets:size(1))
  
  -- select foreground rois
  local control = 0
  local idx = fg_idx[torch.eq(fg_inds,1)]
  while control < fg_num do
    local temp = torch.random(1, idx:size(1))
    if(idx[temp] == 0) then
      idx[temp]= 1
      control = control + 1
    end
  end
  fg_idx[torch.eq(fg_inds,1)] = idx
  -- number of negative samples
    -- assign the labels to all positive boxes
  labels = torch.Tensor(bbox_targets:size(1), 1):fill(3)
  labels[torch.eq(fg_idx, 1)] = 1
  
  labels_weight = torch.zeros(bbox_targets:size(1), 1)
  labels_weight[torch.eq(fg_idx, 1)] = 1
  
  -- same for bg if there are any background rois
  if ( bg_num > 0) then
    bg_idx = torch.zeros(bbox_targets:size(1), 1)
    
    -- select background rois
    control = 0
    idx = bg_idx[torch.eq(bg_inds,1)]
    
    while control < bg_num do
      local temp = torch.random(1, idx:size(1))
      if(idx[temp] == 0) then
        idx[temp]= 1
        control = control + 1
      end
    end
    bg_idx[torch.eq(bg_inds,1)] = idx
    
    labels[torch.eq(bg_idx, 1)] = 2
    labels_weight[torch.eq(bg_idx, 1)] = conf.bg_weights

  end
  
  -- loss for positive targets is 1, rest zero
  bbox_targets = bbox_targets[{{},{2,5}}]
  local bbox_loss_weights = torch.mul(bbox_targets, 0)
  fg_idx = fg_idx:cat(fg_idx,2):cat(fg_idx,2):cat(fg_idx,2)
  bbox_loss_weights[torch.eq(fg_idx, 1)] = 1
  return labels, labels_weight ,bbox_targets, bbox_loss_weights
end
