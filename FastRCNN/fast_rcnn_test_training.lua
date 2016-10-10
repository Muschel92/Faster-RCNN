

function test_training()
  model:evaluate()
    
  local perm = torch.randperm(#image_roidb_train)
  perm = perm[{1,12}]:split(2)
  
  for i = 1,#perm do
    local im_list = {}
    local im1 = image.load(image_roidb_train[perm[i][1]].path, 3, 'byte')
    local scale1 = 0
    im1, scale1 = prep_im_for_blob(im1, conf.image_mean, 600, 1000)
    
    local ex_boxes1 = image_roidb_train[perm[i][1]].ex_boxes[{1,300}]
    local s_fact_1 = scale1:div(image_roidb_train[perm[i][1]].size)
    ex_boxes1 = scale_rois(ex_boxes1, s_fact_1)
    local ex_feat_1 = map_im_rois_to_feat_map(image_roidb_train[perm[i][1]], ex_boxes1, 1, scale1)
    local nr_1 = torch.ones(300,1)
    ex_feat_1 = nr_1:cat(ex_feat_1)
    
    local im2 = image.load(image_roidb_train[perm[i][2]].path, 3, 'byte')
    local scale2 = 0
    im2, scale2 = prep_im_for_blob(im1, conf.image_mean, 600, 1000)
    
    local ex_boxes2 = image_roidb_train[perm[i][2]].ex_boxes[{1,300}]
    local s_fact_2 = scale1:div(image_roidb_train[perm[i][2]].size)
    ex_boxes2 = scale_rois(ex_boxes2, s_fact_2)
    local ex_feat_2 = map_im_rois_to_feat_map(image_roidb_train[perm[i][2]], ex_boxes2, 1, scale2)
    local nr_2 = torch.ones(300,1)
    ex_feat_2 = nr_2:cat(ex_boxes2)
    
    local ex_feat = ex_feat_1:cat(ex_feat_2, 1)
    
    table.insert(im_list, im1)
    table.insert(im_list, im2)
    
    local blob = im_list_to_blob(im_list):cuda() 
      
    table.insert(mini_batch, blob:cuda())
    table.insert(mini_batch, ex_feat:cuda())
    
    output = model:forward(mini_batch)
    
    local y_labels = torch.max(output[1], 2)
    
    local positives = y_labels:lt(21)
    local nr_pos = torch.sum(positives)
    
    if nr_pos > 0 then
      
    
    end
end
