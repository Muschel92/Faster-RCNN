-- calculates the bboxes from the regression
-- input: roidb_information (feature_map_size),
-- image_size which is fed into the network
-- index of used scale in conf.scales
-- the calculated regression
function calculate_bbox_from_reg_output(image_roidb, size, scale, reg, mean, stds)
  
  local anchors = proposal_locate_anchors_single_scale(size, conf, conf.scales[scale], image_roidb.feature_map_size[scale])
  local ex_boxes = bbox_from_regression(anchors:cuda(), reg, mean, stds)
  
  return torch.round(ex_boxes )
  
end
