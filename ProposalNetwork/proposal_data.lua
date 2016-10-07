conf.image_means =  torch.load(conf.data_path .. 'meanImage.t7')
    
conf.anchors = proposal_generate_anchors(conf)

if(conf.calc_feature_map) then

  feature_map = torch.load(conf.network_path .. conf.feature_map)

  print('==> Calculating feature_map sizes TRAIN')
        
  imdb_train = torch.load(conf.data_path .. 'train_roidb.t7')
  imdb_train = calculate_feature_map_size(conf, imdb_train, feature_map)
  torch.save(conf.data_path .. 'train_roidb.t7', imdb_train)
  
  print('==> Calculating bboxes TRAIN')
  
  image_roidb_train, mean_boxes, stds_boxes = proposal_prepare_image_roidb(conf, imdb_train)
  
  torch.save(conf.data_path .. 'train_roidb_all.t7', image_roidb_train)  
  torch.save(conf.data_path .. 'mean_bboxes.t7', mean_boxes)
  torch.save(conf.data_path .. 'stds_bboxes.t7', stds_boxes)
  
  if conf.do_validation then 
    print('==> Calculating feature_map sizes VAL')

    imdb_val = torch.load(conf.data_path .. 'val_roidb.t7')
    imdb_val = calculate_feature_map_size(conf, imdb_val, feature_map)
    torch.save(conf.data_path .. 'val_roidb.t7', imdb_val)
    
    print('==> Calculating bboxes VAL')

    
    image_roidb_val, val_mean, val_stds = proposal_prepare_image_roidb(conf, imdb_val)
    
    torch.save(conf.data_path .. 'val_roidb_all.t7', image_roidb_val)
    torch.save(conf.data_path .. 'mean_val.t7', val_mean)
    torch.save(conf.data_path .. 'stds_val.t7', val_stds)
  end

  
elseif (conf.prep_image_roidb) then
  imdb_train = torch.load(conf.data_path .. 'train_roidb.t7')
    
  print('==> Calculating bboxes')
  
  image_roidb_train, mean_boxes, stds_boxes = proposal_prepare_image_roidb(conf, imdb_train)
  
  torch.save(conf.data_path .. 'train_roidb_all.t7', image_roidb_train)  
  torch.save(conf.data_path .. 'mean_bboxes.t7', mean_boxes)
  torch.save(conf.data_path .. 'stds_bboxes.t7', stds_boxes)
  
  if conf.do_validation then 

    imdb_val = torch.load(conf.data_path .. 'val_roidb.t7')
    
    print('==> Calculating bboxes VAL')

    
    image_roidb_val, val_mean, val_stds = proposal_prepare_image_roidb(conf, imdb_val)
    
    torch.save(conf.data_path .. 'val_roidb_all.t7', image_roidb_val)
    torch.save(conf.data_path .. 'mean_val.t7', val_mean)
    torch.save(conf.data_path .. 'stds_val.t7', val_stds)
  end
  

else
  print("==> Loading Training Data")
  image_roidb_train = torch.load(conf.data_path .. 'train_roidb_all.t7')

  if conf.do_validation then
    image_roidb_val = torch.load(conf.data_path .. 'val_roidb_all.t7')
      
    val_mean = torch.load(conf.data_path .. 'mean_val.t7')
    val_stds = torch.load(conf.data_path .. 'stds_val.t7')
    
    val_mean = val_mean:cuda()
    val_stds = val_stds:cuda() 
    
  end

  mean_boxes = torch.load(conf.data_path .. 'mean_bboxes.t7')
  stds_boxes = torch.load(conf.data_path .. 'stds_bboxes.t7')
  
  mean_boxes = mean_boxes:cuda()
  stds_boxes = stds_boxes:cuda()
  
end

if(conf.testing == true and conf.test_set == 'test') then
  print(' ==> Loading Test Data')
  image_roidb_test = torch.load(conf.data_path .. 'test_roidb_all.t7')
end

print(' ==> Loaded Data')