
test_set = {}
im_blob = torch.CudaTensor()
labels_prop = torch.CudaTensor()
regression = torch.CudaTensor()
ex_boxes = torch.CudaTensor()

softmax = nn.SoftMax():cuda()
function test_for_fast_rcnn()
  model:evaluate()
  
  local tic = torch.tic()
  
  if(conf.test_set == 'train') then
    test_set = image_roidb_train
    
    local results = test_batch(test_set, mean_boxes, stds_boxes)
    
    print('Time to complete testing: ' .. torch.toc(tic))
    
    return results

  elseif(conf.test_set == 'trainval') then
    --test_set = image_roidb_val

    local results_train = test_batch(image_roidb_train,  mean_boxes, stds_boxes)
    local results_val = test_batch(image_roidb_val, val_mean, val_stds)
    
    local results = {}
    
    table.insert(results, results_train)
    table.insert(results, results_val)
    
    print('Time to complete testing: ' .. torch.toc(tic))
  
    return results
    
  elseif (conf.test_set =='val') then
    test_set = image_roidb_val
    
    local results = test_batch(test_set, val_mean, val_stds)
    
    print('Time to complete testing: ' .. torch.toc(tic))
    
    return results

  elseif (conf.test_set == 'test') then
    test_set = image_roidb_test
    
     local results = test_batch(test_set, val_mean, val_stds)
    
    print('Time to complete testing: ' .. torch.toc(tic))
    
    return results
  end
  
    local results = test_batch(test_set)
    
    print('Time to complete testing: ' .. torch.toc(tic))
    
    return results
  end

function test_batch(test_set, mean, stds)
  local results = {}
  
  for i = 1, #test_set do
    for j = 1, conf.scales:size(1) do
      im = image.load(test_set[i].path, 3, 'byte')
      
      im_blob, scale = prep_im_for_blob(im, conf.image_means, conf.scales[1], conf.max_size)
      
      -- Switch from rgb to bgr
      im_blob:index(1, torch.LongTensor{3,2,1})
      
      output = model:forward(im_blob:cuda())
      
      -- get labels as bboxes x 2 (pos and neg score)
      labels_prop = output[1]:permute(2,3,1)
      labels_prop = labels_prop:reshape(labels_prop:size(1)*labels_prop:size(2)*labels_prop:size(3)/3, 3)
      labels_prop = labels_prop[{{},{1,2}}]
      labels_prop = softmax:forward(labels_prop)
      
      --print(output)
      regression = output[2]:permute(2,3,1)
      regression = regression:reshape(regression:size(1)*regression:size(2)*regression:size(3)/4, 4)
          
      local im_size = torch.Tensor{im_blob:size(2), im_blob:size(3)}
        
      -- calculate anchors
      ex_boxes = calculate_bbox_from_reg_output(test_set[i], im_size, 1, regression, mean, stds)
      
      local out = {}
      
      table.insert(out, labels_prop[{{},1}])
      table.insert(out, ex_boxes)
      table.insert(out, scale)
      
      table.insert(results, out)   
    end
  end
  
  return results
  
end

