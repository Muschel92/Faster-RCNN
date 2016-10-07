
-- creates Tensor with all images
-- [numberOfImages x 3 x max_height x max_width]
-- all images start at [1,1,:,:]
function im_list_to_blob(ims)
  im = ims
  
  local max_shape_1 = 0
  local max_shape_2 = 0
  -- find the greatest width and hight
  for i = 1, #ims do
    if(ims[i]:size(2) > max_shape_1) then
      max_shape_1 = ims[i]:size(2)
    end
    if(ims[i]:size(3) > max_shape_2) then
      max_shape_2 = ims[i]:size(3)
    end
  end
  
  blob = torch.Tensor(#ims, 3, max_shape_1, max_shape_2):fill(0)
  for i = 1,#ims do
    blob[i][{{}, {1,ims[i]:size(2)},{1, ims[i]:size(3)}}]:copy(ims[i])
  end
  
  if torch.sum(blob:ne(blob)) > 0 then
    repl()
  end

  
  return blob
end
