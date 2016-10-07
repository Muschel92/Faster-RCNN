-- scales the roidbs from one scale back to the image scale
-- restricts the border of the rois to the image size
function restrict_rois_to_image_size(ex_boxes, image_size)
  
  ex_boxes[{{},1}][ex_boxes[{{},1}]:lt(1)] = 1  
  ex_boxes[{{},3}][ex_boxes[{{},3}]:gt(image_size[1])] = image_size[1]
  ex_boxes[{{},3}][ex_boxes[{{},3}]:lt(1)] = 1
  ex_boxes[{{},1}][ex_boxes[{{},1}]:gt(ex_boxes[{{},3}])] = ex_boxes[{{},3}][ex_boxes[{{},1}]:gt(ex_boxes[{{},3}])]
  ex_boxes[{{},3}][ex_boxes[{{},3}]:lt(ex_boxes[{{},1}])] = ex_boxes[{{},1}][ex_boxes[{{},3}]:lt(ex_boxes[{{},1}])]
  
  ex_boxes[{{},2}][ex_boxes[{{},2}]:lt(1)] = 1
  ex_boxes[{{},4}][ex_boxes[{{},4}]:gt(image_size[2])] = image_size[2]
  ex_boxes[{{},4}][ex_boxes[{{},4}]:lt(1)] = 1 
  ex_boxes[{{},2}][ex_boxes[{{},2}]:gt(ex_boxes[{{},4}])] = ex_boxes[{{},4}][ex_boxes[{{},2}]:gt(ex_boxes[{{},4}])]  
  ex_boxes[{{},4}][ex_boxes[{{},4}]:lt(ex_boxes[{{},2}])] = ex_boxes[{{},2}][ex_boxes[{{},4}]:lt(ex_boxes[{{},2}])]
  
  return ex_boxes
end
