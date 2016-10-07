

-- transform the scaled rois from image size to feature map size
function map_im_rois_to_feat_map(roidb, rois, im_scale, scales)
    
  local scale = torch.cdiv(roidb.feature_map_size[im_scale]:float(), scales)
  local new_rois = torch.CudaTensor(rois:size())
  
  new_rois[{{},1}] = (rois[{{},1}] * scale[1])
  new_rois[{{},2}] = (rois[{{},2}] * scale[2])
  new_rois[{{},3}] = (rois[{{},3}] * scale[1])
  new_rois[{{},4}] = (rois[{{},4}] * scale[2])
  
  new_rois = torch.ceil(new_rois)
  new_rois[{{},3}][new_rois[{{},3}]:gt(roidb.feature_map_size[im_scale][1])]  = roidb.feature_map_size[im_scale][1]
  new_rois[{{},4}][new_rois[{{},4}]:gt(roidb.feature_map_size[im_scale][2])]  = roidb.feature_map_size[im_scale][2]
  
  return new_rois 
  
end
