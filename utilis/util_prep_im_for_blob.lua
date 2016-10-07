-- substracts the mean from the image
-- and rescales it to the target_size if possible
function prep_im_for_blob(im, im_means, target_size, max_size)

  if(type(im_means) == "number") then
      im_means = torch.zeros(3):byte()
  end
  
  -- calculates new width, height of image
  local size = torch.Tensor{im:size(2), im:size(3)}
  
  local im_scale = prep_im_for_blob_size(size, target_size, max_size)

  im = image.scale(im, im_scale[2], im_scale[1])  


  im[1]:add(-im_means[1])
  im[2]:add(-im_means[2])
  im[3]:add(-im_means[3])
  
  return im, im_scale
  
  
end
