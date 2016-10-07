
function calc_rois_on_feature_map(rois, im_size, feature_map_size)
  local scale = torch.cdiv(feature_map_size, im_size)
  local new_rois = torch.Tensor(rois:size())
  
  new_rois[{{},1}]:cmul(rois[{{},1}], scale[1]):round()
  new_rois[{{},2}]:cmul(rois[{{},2}], scale[2]):round()
  new_rois[{{},3}]:cmul(rois[{{},3}], scale[1]):round()
  new_rois[{{},4}]:cmul(rois[{{},4}], scale[2]):round()
  
  return new_rois  
end
