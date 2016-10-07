
require 'torch'
require 'nn'
require 'paths'
require 'cudnn'
require 'cunn'
require 'cutorch'
require 'image'
require 'optim'

color = require 'trepl.colorize'
-----------------------------------------------------------------------------
dofile('main_config.lua')
dofile('ProposalNetwork/proposal_generate_anchors.lua')
dofile('ProposalNetwork/proposal_locate_anchors.lua')
dofile('ProposalNetwork/proposal_prepare_image_roidb.lua')
dofile('ProposalNetwork/proposal_generate_mini_batch.lua')
dofile('ProposalNetwork/proposal_generate_batch_roidbs.lua')
dofile('utilis/util_prep_im_for_blob_size.lua')
dofile('utilis/util_flip_rois.lua')
dofile('utilis/util_prep_im_for_blob.lua')
dofile('utilis/util_im_list_to_blob.lua')
dofile('utilis/util_fast_rcnn_bbox_transform.lua')
dofile('utilis/util_generate_mini_batch_files.lua')
dofile('utilis/util_boxoverlap.lua')
dofile('utilis/util_scale_rois.lua')
dofile('utilis/util_bbox_from_regression.lua')
dofile('utilis/util_restric_rois_image_size.lua')
dofile('utilis/util_calculate_bbox_from_reg_output.lua')
dofile('utilis/util_img_from_mean.lua')
dofile('Preprocessing/preparation_cal_feature_map_size.lua')

torch.setdefaulttensortype('torch.FloatTensor')

conf = config()

torch.setnumthreads(conf.threads)

assert(cutorch.getDeviceCount() == conf.nGPU, 'Make GPUs invisible! - export CUDA_VISIBLE_DEVICES=0,1,2')

-- wichtig: macht schnell
cudnn.benchmark = true
cudnn.fastest = true

torch.setnumthreads(conf.threads)
torch.manualSeed(conf.rng_seed)

cutorch.setDevice(conf.defGPU)

print '==> executing all'

dofile 'main_util.lua'
--dofile 'main_data.lua'

dofile 'ProposalNetwork/proposal_logger.lua'
dofile 'ProposalNetwork/proposal_data.lua'
dofile 'ProposalNetwork/proposal_model.lua'

if conf.testing then 
  
  if conf.train_stage == 'rpn_1' then
    dofile 'ProposalNetwork/proposal_test_rpn1.lua'
  else 
    dofile 'ProposalNetwork/proposal_test_rpn2.lua'
  end
  local results = test_for_fast_rcnn()
  if(conf.test_set == 'trainval') then  
    torch.save(conf.test_path .. 'train_results_' .. conf.model_type, results[1])
    torch.save(conf.test_path .. 'val_results_' .. conf.model_type, results[2])
  else
    torch.save(conf.test_path .. 'results_' .. conf.model_type, results)
  end
  
    
else
  if conf.train_stage == 'rpn_1' then
    dofile 'ProposalNetwork/proposal_train_stage1.lua'
    if conf.do_validation then
      dofile 'ProposalNetwork/proposal_validation_stage1.lua'
    end
  else 
    dofile 'ProposalNetwork/proposal_train_stage2.lua'
    if conf.do_validation then
      dofile 'ProposalNetwork/proposal_validation_stage2.lua'
    end
  end
  
  if (conf.load_old_network == true) then
    epoch = trainState.epoch
  else
    epoch = 1
  end
  
  --writeReport(tmp)
  print '==> training!'
  while true do   
    --proposal_train()
    train()
    
    if conf.do_validation then
      validation()
    end
    
    logging()
    writeReport()
    epoch = epoch+1
    if(conf.learningRate ~= 0) then
      if(epoch % conf.epoch_step == 0) then
        optimState.learningRate = optimState.learningRate*conf.gamma
      end
    end
    if(epoch == 10 * conf.epoch_step) then
      break
    end
  end
end

