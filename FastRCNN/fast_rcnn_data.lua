
conf.image_means =  torch.load(conf.data_path .. 'meanImage.t7')
--conf.image_means:div(255)

if(conf.prep_image_roidb) then
  imdb_train = torch.load(conf.data_path .. 'train_roidb.t7')
  --imdb_val= torch.load(conf.data_path .. 'val_roidb.t7')
  
  print('==> Calculating bboxes')
  
  image_roidb_train, mean_boxes, stds_boxes = fast_rcnn_prepare_image_roidb(conf, imdb_train)
  --image_roidb_val, val_mean, val_stds = fast_rcnn_prepare_image_roidb(conf, imdb_val)
  
  torch.save(conf.data_path .. 'train_roidb_all.t7', image_roidb_train)
  --torch.save(conf.data_path .. 'val_roidb_all.t7', image_roidb_val)
  torch.save(conf.data_path .. 'mean_bboxes.t7', mean_boxes)
  --torch.save(conf.data_path .. 'mean_val.t7', val_mean)
  torch.save(conf.data_path .. 'stds_bboxes.t7', stds_boxes)
  --torch.save(conf.data_path .. 'stds_val.t7', val_stds)
  
else
  print("==> Loading Training Data")
  image_roidb_train = torch.load(conf.data_path .. 'train_roidb_all.t7')
  --image_roidb_val = torch.load(conf.data_path .. 'val_roidb_all.t7')
  
  mean_boxes = torch.load(conf.data_path .. 'mean_bboxes.t7')
  stds_boxes = torch.load(conf.data_path .. 'stds_bboxes.t7')
  --val_mean = torch.load(conf.data_path .. 'mean_val.t7')
  --val_stds = torch.load(conf.data_path .. 'stds_val.t7')
  
  mean_boxes = mean_boxes:cuda()
  stds_boxes = stds_boxes:cuda()
  --val_mean = val_mean:cuda()
  --val_stds = val_stds:cuda()  
end

if(conf.testing == true and conf.test_set == 'test') then
  image_roidb_test = torch.load(conf.data_path .. 'test_roidb_all.t7')
end