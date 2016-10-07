
-- scales the roidbs from one scale back to the image scale
-- restricts the border of the rois to the image size
function restrict_roidb_to_image_size(roidbs, ex_boxes, scale, target_scale)
  
  for i = 1, #roidbs do
    
    local ex_box_scale = prep_im_for_blob_size(roidbs[i].size, scale, target_scale) 
    ex_boxes[i][2] = scale_rois(ex_boxes[i][2]:float(), ex_box_scale:float(), roidbs[i].size:float())  
    
    ex_boxes[i][2][{{},1}][ex_boxes[i][2][{{},1}]:lt(1)] = 1  
    ex_boxes[i][2][{{},3}][ex_boxes[i][2][{{},3}]:gt(roidbs[i].size[1])] = roidbs[i].size[1]
    ex_boxes[i][2][{{},3}][ex_boxes[i][2][{{},3}]:lt(1)] = 1
    ex_boxes[i][2][{{},1}][ex_boxes[i][2][{{},1}]:gt(ex_boxes[i][2][{{},3}])] = ex_boxes[i][2][{{},3}][ex_boxes[i][2][{{},1}]:gt(ex_boxes[i][2][{{},3}])]
    ex_boxes[i][2][{{},3}][ex_boxes[i][2][{{},3}]:lt(ex_boxes[i][2][{{},1}])] = ex_boxes[i][2][{{},1}][ex_boxes[i][2][{{},3}]:lt(ex_boxes[i][2][{{},1}])]
    
    ex_boxes[i][2][{{},2}][ex_boxes[i][2][{{},2}]:lt(1)] = 1
    ex_boxes[i][2][{{},4}][ex_boxes[i][2][{{},4}]:gt(roidbs[i].size[2])] = roidbs[i].size[2]
    ex_boxes[i][2][{{},4}][ex_boxes[i][2][{{},4}]:lt(1)] = 1
    ex_boxes[i][2][{{},2}][ex_boxes[i][2][{{},2}]:gt(ex_boxes[i][2][{{},4}])] = ex_boxes[i][2][{{},4}][ex_boxes[i][2][{{},2}]:gt(ex_boxes[i][2][{{},4}])]  
    ex_boxes[i][2][{{},4}][ex_boxes[i][2][{{},4}]:lt(ex_boxes[i][2][{{},2}])] = ex_boxes[i][2][{{},2}][ex_boxes[i][2][{{},4}]:lt(ex_boxes[i][2][{{},2}])]
  end
  
  return ex_boxes
end
