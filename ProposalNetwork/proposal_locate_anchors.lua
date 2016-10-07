
-- returns the anchors of the images as table: [scales][anchors]
function proposal_locate_anchors(conf, im_size, feature_map_size) 
    
    local anchors = {}
    local im_scales = {}
    
    for i = 1, conf.scales:size(1) do
      anchor, im_scale = proposal_locate_anchors_single_scale(im_size, conf, conf.scales[i], feature_map_size[i])
      table.insert(anchors, anchor)
      table.insert(im_scales, im_scale)
    end
   
    return anchors, im_scales
end

-- returns the scaling factor of the image for the target_scale
-- [(1 scale: (1 ratios for all anchor positions)x ratios) x scales]
function proposal_locate_anchors_single_scale(im_size, conf, target_scale, feature_map_size)
    
    -- calculate height, width of scaled image
    im_scale = prep_im_for_blob_size(im_size, target_scale, conf.max_size)
        
    -- calculate all the shifts in x and y direction
    shift_h = torch.range(1, feature_map_size[1]):resize(feature_map_size[1], 1):mul(conf.feat_stride[1])
    shift_w = torch.range(1, feature_map_size[2]):resize(1, feature_map_size[2]):mul(conf.feat_stride[2])
    size_h = shift_h:size(1)
    size_w = shift_w:size(2)
    
    -- calculate all the possible combinations of shifts in x and y direction of the base_anchor
    --[(1x x (all y)) x all x]
    h = shift_h:repeatTensor(1, size_w):resize(size_h * size_w)
    w = shift_w:repeatTensor(size_h, 1):resize(size_h * size_w)
      
      
    -- Calculate all possible shifts for each  base_anchor
    size_ratio = conf.anchors:size(1)
    size_scale = conf.anchors:size(2)
    
    -- for every h: w_1 to w_n
    anchor_c = h:cat(w,2):cat(h,2):cat(w,2)
    
    s = conf.anchors:size(1)
    size = anchor_c:size(1)

    anchors = torch.Tensor(anchor_c:size(1) * s, 4)
  
    for i = 1, anchor_c:size(1) do
        for j = 1, conf.anchors:size(1) do 
          anchors[{{(i-1)*s + j}}] = torch.add(conf.anchors[j] ,anchor_c[i])
        end
    end

    return anchors, im_scale
end

function meshgrid (vector1, vector2)
  local size1 = vector1:size(1)
  local size2 = vector2:size(1)
  
  local meshgrid1 = torch.Tensor(size1, size2)
  local meshgrid2 = torch.Tensor(size1, size2)
  
  for i = 1, size2 do
      meshgrid1[{{}, i}] = vector1
  end
  
  for i = 1, size1 do
      meshgrid2[{i, {}}] = vector2
  end

  
  return meshgrid1, meshgrid2
end

