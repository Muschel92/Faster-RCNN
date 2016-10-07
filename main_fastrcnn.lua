
require 'torch'
require 'nn'
require 'paths'
require 'cudnn'
require 'cunn'
require 'cutorch'
require 'image'
require 'optim'
require 'inn'

color = require 'trepl.colorize'
-----------------------------------------------------------------------------
dofile('utilis/util_boxoverlap.lua')
dofile('utilis/util_scale_rois.lua')
dofile('utilis/util_fast_rcnn_bbox_transform.lua')
dofile('utilis/util_prep_im_for_blob_size.lua')
dofile('utilis/util_im_list_to_blob.lua')
dofile('utilis/util_prep_im_for_blob.lua')
dofile('utilis/util_bbox_from_regression.lua')
dofile('utilis/util_calculate_bbox_from_reg_output.lua')
dofile('utilis/util_img_from_mean.lua')
dofile('utilis/util_generate_mini_batch_files.lua')
dofile('utilis/util_generate_batch_roidbs.lua')
dofile('utilis/util_flip_rois.lua')
dofile('Preprocessing/preparation_cal_feature_map_size.lua')
dofile('FastRCNN/fast_rcnn_prepare_image_roidb.lua')
dofile('FastRCNN/fast_rcnn_map_im_rois_to_feat_map.lua')
dofile('FastRCNN/fast_rcnn_config.lua')

torch.setdefaulttensortype('torch.FloatTensor')

conf = config()

torch.setnumthreads(conf.threads)

assert(cutorch.getDeviceCount() == conf.nGPU, 'Make GPUs invisible! - export CUDA_VISIBLE_DEVICES=0,2')

-- wichtig: macht schnell
cudnn.benchmark = true
cudnn.fastest = true

torch.setnumthreads(conf.threads)
torch.manualSeed(conf.rng_seed)

cutorch.setDevice(conf.defGPU)

print '==> executing all'

dofile 'main_util.lua'
dofile 'FastRCNN/fast_rcnn_data.lua'
dofile 'FastRCNN/fast_rcnn_model.lua'

print('==> finished loading model')
dofile ('FastRCNN/fast_rcnn_generate_mini_batch.lua')
dofile('utilis/util_restric_rois_image_size.lua')
dofile('FastRCNN/fast_rcnn_logger.lua')

if conf.testing then 
    while true do   
      --proposal_train()
      train()
      validation()
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
else
	dofile('FastRCNN/fast_rcnn_train.lua')
  --dofile('FastRCNN/fast_rcnn_validate.lua')

	print '==> training!'
	while true do   
		train()
    if conf.do_validation then
      validation()
    end
    
		logging()
		writeReport()
		epoch = epoch+1
		if(conf.learningRate ~= 0) then
			if(epoch % conf.epoch_step == 0) then
			optimState.learningRate = optimState.learningRate * conf.gamma
			end
		end
		if(epoch == conf.max_iter) then
			break
		end
	end
end

