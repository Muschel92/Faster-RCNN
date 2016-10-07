function generate_batch_roidbs(files, images)
  
  ims = {}
  
  for i = 1,files:size(1) do
    table.insert(ims, images[files[i]])
  end
  
  return ims
  
end  
