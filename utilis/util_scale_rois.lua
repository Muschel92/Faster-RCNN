-- scales a bbox according to a new scale of an image
-- im_scale: new_height, new_width
-- im_size: original scale
-- returns rois with new scale
function scale_rois(gt, im_size, im_scale)

  local rois = gt
  scaling_factor = torch.cdiv(im_scale, im_size)

  scaled_rois = torch.Tensor(rois:size(1), 4):cuda()
  
  scaled_rois[{{},1}] = torch.round(torch.mul(rois[{{},1}], scaling_factor[1]))
  scaled_rois[{{},2}] = torch.round(torch.mul(rois[{{},2}], scaling_factor[2]))  
  scaled_rois[{{},3}] = torch.round(torch.mul(rois[{{},3}], scaling_factor[1]))
  scaled_rois[{{},4}] = torch.round(torch.mul(rois[{{},4}], scaling_factor[2]))

  return (scaled_rois)
end