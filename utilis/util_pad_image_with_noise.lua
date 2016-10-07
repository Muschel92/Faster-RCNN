
-- pads an image with random noise to size height x width
function pad_image_with_noise(image, widht, height)
    local im_width = image:size()[3]
    local im_height = image:size()[2]
    
    local pad_im = torch.rand(3, height, width) * 256
    pad_im = pad_im:byte()
    
    local pad_width = width - im_width
    local pad_height = height - im_height
    
    local ymin = math.floor(pad_height/2 +0.5)
    local xmin = math.floor(pad_width/2 + 0.5)
    local ymax = im_height - math.ceil(pad_height/2 + 0.5)
    local xmax = im_width - math.floor(pad_width/2 + 0.5)
    
    pad_im[{{},{ymin, ymax}, {xmin, xmax}] = image
      
    return pad_im
    
  
end
