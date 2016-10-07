function generate_mini_batch_files(batchsize, images)
  
  if (batchsize > #images) then
    batchsize = #images
  end
  
  local indices = torch.randperm(#images)      
  local ims = {}
    
  for i = 1,batchsize do
    table.insert(ims, images[indices[i]])
  end
  
  return ims
  
end  
