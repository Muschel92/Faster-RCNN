-- script to transform the roidbs of the rpn net to the roidbs of the fast rcnn net
-- need the results of the rpn network


require('torch')
require('cudnn')
require('cunn')

dofile('utilis/util_rpn_results_to_fast_rcnn.lua')
dofile('utilis/util_nms.lua')
dofile('utilis/util_boxoverlap.lua')
dofile('utilis/util_restrict_roidb_to_image_size.lua')
dofile('utilis/util_scale_rois.lua')
dofile('utilis/util_prep_im_for_blob_size.lua')

torch.setdefaulttensortype('torch.FloatTensor')

cudnn.benchmark = true
cudnn.fastest = true

db_train = torch.load('/data/ethierer/ObjectDetection/FasterRCNN/Data/ProposalNetwork/Resnet_50/VOC2007TrainVal/train_roidb_all.t7')
--db_val = torch.load('/data/ethierer/ObjectDetection/FasterRCNN/Data/ProposalNetwork/Data_For_Debug/val_roidb_all.t7')

train_results= torch.load('/data/ethierer/ObjectDetection/FasterRCNN/Results/rpn_2/Res50_Sat1Okt_ep16.t7/SunOct209:07:102016/results_Res50_Sat1Okt_ep16.t7')
--val_results = torch.load('/data/ethierer/ObjectDetection/FasterRCNN/Results/rpn_1/VGG16_ZeroBias_512_TrainVal_2208_ep15.t7/MonAug2909:44:032016/val_results_VGG16_ZeroBias_512_TrainVal_2208_ep15.t7')

train_results = restrict_roidb_to_image_size(db_train, train_results, 600, 1000)
--val_results = restrict_roidb_to_image_size(db_val, val_results, 600, 1000)
collectgarbage()

print('restricted image size') 

roidbs_train = transform_rpn_data_to_fast_rcnn(db_train, train_results)
--roidbs_val = transform_rpn_data_to_fast_rcnn(db_val, val_results)

torch.save('/data/ethierer/ObjectDetection/FasterRCNN/Data/FastRCNN/Res50/VOC2007TrainVal_noGt/train_roidb.t7', roidbs_train)
--torch.save('/data/ethierer/ObjectDetection/FasterRCNN/Data/FastRCNN/Data_For_Debug/val_roidb_all.t7', roidbs_val)