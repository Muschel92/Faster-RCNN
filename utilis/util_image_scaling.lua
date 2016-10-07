
-- calculates the padding for the image and adds it to the groundtruth boxes
function preparation_image_scaling(prep_size, image_roidb)
        
    for i, image in ipairs(image_roidb) do
        local im_size = im.size
        
        local im_width = image.size[2]
        local im_height = image.size[1]
        
        local pad_width = width - im_width
        local pad_height = height - im_height
        
        local ymin = math.floor(pad_height/2 +0.5)
        local xmin = math.floor(pad_width/2 + 0.5)
        
        local padding = torch.Tensor{xmin, ymin, xmin, ymin}
        
        for j = 1, image_roidb[i].boxes:size(1) do
            image_roidb[i].boxes[j][{{}}] = padding        
    end
 
    return image_roidb
end
