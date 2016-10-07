
-- calculates the new width and height of the image according to the target_size
-- smallest dimension of image is scaled to target size
-- returns new [height, width]
function prep_im_for_blob_size(im_size, target_size, max_size)
    local im_size_min = math.min(im_size[1], im_size[2])
    local im_size_max = math.max(im_size[1], im_size[2])
    local im_scale = target_size /im_size_min
    
    if (math.floor(im_scale * im_size_max + 0.5) > max_size) then
        im_scale = max_size / im_size_max
    end
    
    im_scale = torch.round(torch.Tensor{im_size[1]*im_scale, im_size[2]*im_scale})
    return (im_scale)
end
