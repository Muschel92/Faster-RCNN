
function calculate_feature_map_size(conf, image_roidb, model) 
       
  model:cuda()     
  for i, ims in ipairs(image_roidb) do

    local feature_map_size = {}
    
    for j = 1, conf.scales:size()[1] do

      im_scale = prep_im_for_blob_size(ims.size, conf.scales[j], conf.max_size)      
      temp = torch.Tensor(1, 3, im_scale[1], im_scale[2]):cuda()      
      resp = model:forward(temp)
     
      table.insert(feature_map_size, torch.Tensor{resp:size(3), resp:size(4)})
    end
    
    image_roidb[i].feature_map_size = feature_map_size
  end
    
    
    return image_roidb
end
