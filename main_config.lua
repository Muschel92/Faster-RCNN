require ('torch')

if not opt then
   
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text('Options:')
   cmd:option('-rng_seed', 6, 'random seed')
   -- Options for GPU
    cmd:option('-nDonkeys', 2   , 'number of parallel threads')
    cmd:option('-threads', 2   , 'number of parallel threads')
    cmd:option('-defGPU', 1, 'Default preferred GPU')
    cmd:option('-nGPU', 1, 'Number of GPUs to be used')
    -- Options RPN or FastRcnn
    cmd:option('-train_stage', 'rpn_1', 'possible options: rpn_1, rpn_2')
    -- Options for image preparations
    cmd:option('-drop_boxes_runoff_image', true, 'whether to drop the anchors that has      edges outside of image boundary')
    cmd:option('-anchor_scales', torch.Tensor{ {1,2},{1,1},{2,1}} , 'ratios list of anchors')
    cmd:option('-anchor_ratios', torch.Tensor{16384,65536, 262144 } , 'scale list of anchors')
    cmd:option('-total_anchors', 9 , 'total number on anchors: scales * ratios')
    cmd:option('-anchor_base_size', 16 , 'the size of base anchor')
    cmd:option('-fg_fraction', 0.5 , 'Fraction of minibatch that is foreground labeled (class > 1)')
    cmd:option('-rois_per_image', 256 , 'Rois per image in minibatch')  
    cmd:option('-fg_thresh', 0.7 , 'Overlap threshold for a ROI to be considered foreground')
    cmd:option('-bg_weights', 1, 'weight of background samples, when weight of foreground samples is 1.0')
    cmd:option('-bg_thresh_hi', 0.3, 'Overlap threshold for a ROI to be considered background (high)')
    cmd:option('-bg_thresh_lo', 0, 'Overlap threshold for a ROI to be considered background (low)')
    cmd:option('-image_means', 128 , 'mean image in RGB-order')
    cmd:option('-different_image_size', true, 'if images can have different image sizes')
    cmd:option('-feat_stride', torch.Tensor{16, 16}, 'Stride in input image pixels at ROI pooling level(network specific)')
   cmd:option('-target_only_gt', true, 'train proposal target only to labled ground-truths or also include other proposal results')
    cmd:option('-scales', torch.Tensor{600} , 'image scale - the short edge of an input image')
    cmd:option('-max_size', 1000 , 'max pixel size of scaled input image')
    cmd:option('-feature_map_size', torch.Tensor{16,16}, 'for testing')
    cmd:option('-feature_map', 'Res_50_Feature_Map.t7', 'path of feature map model')
    cmd:option('-calc_feature_map', false   , 'true if feature map size needs to be calculated')
    cmd:option('-prep_image_roidb', false , 'true if image roidb preparation needs to be calculated')
    
    -- Options for continuous trainung
    cmd:option('-trainState_name', 'trainState_100.t7' , 'trainState to be loaded')
    cmd:option('-optimState_name', 'optimState_100.t7' , 'optimState to be loaded')
    cmd:option('-load_old_network', false   , 'true if trained network should be loaded') 
    cmd:option('-random_seed_path', 'random_seed.t7'   , 'path to the saved random seed')
    
    -- Options for loading/ saving the model
    cmd:option('-create_network', false  , 'true if new network should be created')

    cmd:option('-network_path', '/data/ethierer/ObjectDetection/FasterRCNN/Model/ProposalNetwork/stage_1/'   , 'path to the networks')
    cmd:option('-model_type', 'VGG16_RPN_Network.t7' , 'name of the model that should be loaded')
    cmd:option('-save_epoch', 2, 'every how many epochs the model is saved')
    
    -- Options for learning
    cmd:option('-do_validation', false, ' if validation should be calculated')
    cmd:option('-max_iter', 12000, 'maximum of batches per epoch')
    cmd:option('-learningRate', 0.0, 'learning rate, zero if a specific shedule in proposal_train.lua is used')
    cmd:option('-weightDecay', 0.0005, 'weight decay')
    cmd:option('-momentum', 0.9, 'momentum')
    cmd:option('-epoch_step', 12, 'nr of epochs when the learning rate decreases by 0.9')
    cmd:option('-gamma', 0.1, 'factor to reduce the learning rate every epoch_step')
    cmd:option('-batch_size', 1 , 'images per batch (only one)')
    cmd:option('-weight_scec', 1, 'weight for the one of the two criteriones (Spatial Cross Entropy)')
    cmd:option('-weight_l1crit', 10, 'weight for the one of the two criteriones (Smooth L1 Criterion)')
    cmd:option('-rois_div', 1, 'number to divide gradients during training (grads / batchsize / rois_div)')
    cmd:option('-smoothL1', 1, 'use smoothL1Criterion or not (MSECriterion)')
    cmd:option('-sizeAverage_cls', 1, 'set sizeAverage = true (1) or false (0) for classification')
    cmd:option('-sizeAverage_reg', 0, 'set sizeAverage = true (1) or false (0) for regression')

    -- Options for saving progress
    
    cmd:option('-save_model_state', '/data/ethierer/ObjectDetection/FasterRCNN/Logger/' , 'Path where to save the current stats of the net')
    
    -- Options for loading/saving data
    cmd:option('-data_path', '/data/ethierer/ObjectDetection/FasterRCNN/Data/ProposalNetwork/VGG16/VOC2007TrainVal/')
      
    -- Option for Testing RPN
    cmd:option('-testing', false   , 'should the framework be tested')
    cmd:option('-test_set', 'train'   , 'set on which should be tested')
    cmd:option('-test_path', '/data/ethierer/ObjectDetection/FasterRCNN/Results/'   , 'set on which should be tested')
    
    -- Options for Fast RCNN
    cmd:option('-bbox_thresh', 0.5, 'threshhold for correct bounding box')
    cmd:option('-fg_thresh_f', 0.5, 'threshhold for correct bounding box')
    cmd:option('-bg_thresh_hi_f', 0.5, 'threshhold for correct bounding box')
    cmd:option('-bg_thresh_lo_f', 0.1, 'threshhold for correct bounding box')
    cmd:option('-rois_per_image_f', 128, 'threshhold for correct bounding box')
    cmd:option('-fg_fraction_f', 0.25, 'threshhold for correct bounding box')
    --cmd:option('-numClasses', 20, 'threshhold for correct bounding box')
   cmd:text()
   opt = cmd:parse(arg or {})
end

function config()
    conf = {}
    conf["train_stage"] = opt.train_stage
    conf["drop_boxes_runoff_image"] = opt.drop_boxes_runoff_image
    conf["scales"] = opt.scales
    conf["max_size"] = opt.max_size
    conf["batch_size"] = opt.batch_size
    conf["fg_fraction"] = opt.fg_fraction
    conf["rois_per_image"] = opt.rois_per_image
    conf["fg_thresh"] = opt.fg_thresh
    conf["bg_weights"] = opt.bg_weights
    conf["bg_thresh_hi"] = opt.bg_thresh_hi
    conf["bg_thresh_lo"] = opt.bg_thresh_lo
    conf["image_means"] = opt.image_means
    conf["different_image_size"] = opt.different_image_size
    conf["feat_stride"] = opt.feat_stride
    conf["target_only_gt"] = opt.target_only_gt
    conf["rng_seed"] = opt.rng_seed
    conf["feature_map_size"] = opt.feature_map_size 
    conf["feature_map"] = opt.feature_map
    conf["anchor_scales"] = opt.anchor_scales
    conf["anchor_ratios"] = opt.anchor_ratios
    conf["total_anchors"] = opt.total_anchors
    conf["anchor_base_size"] = opt.anchor_base_size
    conf["train_list"] = opt.train_list
    conf["val_list"] = opt.val_list
    conf["max_iter"] = opt.max_iter
    conf["do_validation"] = opt.do_validation
    conf["learningRate"] = opt.learningRate
    conf["momentum"] = opt.momentum
    conf["gamma"] = opt.gamma
    conf["weightDecay"] = opt.weightDecay
    conf["weight_scec"] = opt.weight_scec
    conf["weight_l1crit"] = opt.weight_l1crit
    conf["sizeAverage_cls"] = opt.sizeAverage_cls
    conf["sizeAverage_reg"] = opt.sizeAverage_reg
    conf["rois_div"] = opt.rois_div
    conf["smoothL1"] = opt.smoothL1
    conf["epoch_step"] = opt.epoch_step
    conf["save_epoch"] = opt.save_epoch
    conf["image_prep_size"] = opt.image_prep_size
    conf["calc_feature_map"] = opt.calc_feature_map
    conf["prep_image_roidb"] = opt.prep_image_roidb
    conf["save_model_state"] = opt.save_model_state
    conf["data_path"] = opt.data_path
    conf["create_proposal_net"] = opt.create_proposal_net
    conf["load_old_network"] = opt.load_old_network
    conf["network_path"] = opt.network_path
    conf["load_trainState_path"] = opt.load_trainState_path
    conf["trainState_name"] = opt.trainState_name
    conf["optimState_name"] = opt.optimState_name
    conf["model_type"] = opt.model_type
    conf["nDonkeys"] = opt.nDonkeys
    conf["threads"] = opt.threads
    conf["defGPU"] = opt.defGPU
    conf["nGPU"] = opt.nGPU
    conf["random_seed_path"] = opt.random_seed_path
    -- Options for testing RPN
    conf["testing"] = opt.testing
    conf["test_set"] = opt.test_set
    conf["test_path"] = opt.test_path
    -- Options for Fast RCNN
    conf["bbox_thresh"] = opt.bbox_thresh
    conf["fg_thresh_f"] = opt.fg_thresh_f
    conf["bg_thresh_hi_f"] = opt.bg_thresh_hi_f
    conf["bg_thresh_lo_f"] = opt.bg_thresh_lo_f
    conf["numClasses"] = opt.numClasses
    conf["rois_per_image_f"] = opt.rois_per_image_f
    conf["fg_fraction_f"] = opt.fg_fraction_f
    
    if(conf.testing == false) then
      conf.save_model_state = paths.concat(conf.save_model_state, conf.train_stage)
      conf.save_model_state = paths.concat(conf.save_model_state, conf.model_type)
      
    -- add date/time
      conf.save_model_state = paths.concat(conf.save_model_state, '' .. os.date():gsub(' ',''))

      paths.mkdir(conf.save_model_state)
      
      conf.save_model_state = conf.save_model_state .. '/'
      
      paths.mkdir(conf.save_model_state .. 'Images')
      paths.mkdir(conf.save_model_state .. 'Models')
      
      print('Will save at '..conf.save_model_state)
    elseif (conf.testing == true) then
      conf.test_path = paths.concat(conf.test_path, conf.train_stage)
      conf.test_path = paths.concat(conf.test_path, conf.model_type)
      conf.test_path = paths.concat(conf.test_path, '' .. os.date():gsub(' ',''))
      paths.mkdir(conf.test_path)
      
      conf.test_path = conf.test_path .. '/'
      
      print('Testing data will be saved at '.. conf.test_path)
    end
    
    return (conf)
end
