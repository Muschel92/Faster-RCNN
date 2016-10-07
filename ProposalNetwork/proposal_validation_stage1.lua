
local target_labels = torch.CudaTensor()
local target_reg = torch.CudaTensor()
local target_bbox_loss = torch.CudaTensor()
local target_weights = torch.CudaTensor()

local val_batch = proposal_generate_mini_batch(conf, image_roidb_val)

local anchors = {}
local pos_boxes = {}
local idx_ex_gt = {}
local scaled_rois = {}
local imgCount = 0



-- calc anchor positions
for i = 1, #image_roidb_val do
    local im_size = torch.Tensor{val_batch[1][i]:size(2), val_batch[1][i]:size(3)}
    local temp1 = proposal_locate_anchors_single_scale( im_size, conf, conf.scales[val_batch[6][i]], image_roidb_val[i].feature_map_size[val_batch[6][i]])
    table.insert(anchors, temp1)
    
    local temp2 = torch.gt(image_roidb_val[i].bbox_targets[val_batch[6][i]][val_batch[7][i]][{{},1}], 0):cuda()
    local temp4 = temp2:cat(temp2,2):cat(temp2,2):cat(temp2,2)
    table.insert(pos_boxes, temp4)
  

    local temp5 = image_roidb_val[i].bbox_targets[val_batch[6][i]][val_batch[7][i]][{{},1}]
    local temp3 = temp5[temp5:gt(0)]
    table.insert(idx_ex_gt, temp3)
    
    local im_scaled = torch.CudaTensor{val_batch[1][i]:size(2),val_batch[1][i]:size(3)}     
    local scale = prep_im_for_blob_size(image_roidb_val[i].size, torch.min(im_scaled), conf.max_size)

    rois = image_roidb_val[i].boxes
    
    if val_batch[7][i] == 2 then
      rois = flip_rois(rois, image_roidb_val[i].size)
    end
    
    rois = scale_rois(image_roidb_val[i].boxes, image_roidb_val[i].size, scale):cuda()
    table.insert(scaled_rois, rois)
    
    
end


function validation()
    
    imgCount = 12
--conf.batch_size < 16 and conf.batch_size or 16
    local perm = torch.randperm(#image_roidb_val)
    local pos = {}
  
    model:evaluate()
    
    val_fg_acc = 0
    val_bg_acc = 0  
    val_reg_accuracy = 0
    val_reg_correct = 0
    val_loss = 0
    
     local tic = torch.tic()
    
    for i = 1, #val_batch[1] do
       -- get target
      target_labels = val_batch[2][i]:cuda()
      target_reg = val_batch[4][i]:cuda()
      target_bbox_loss = val_batch[5][i]:cuda()
      target_weights = val_batch[3][i]:cuda()
      
      -- calc net output
      local y = model:forward(val_batch[1][i]:cuda())
      
      -- transform output boxes to correct dimensions
      local k = conf.total_anchors
      local y_reg = y[2]:permute(2,3,1)
      y_reg = y_reg:reshape(y_reg:size(1)*y_reg:size(2)*k, 4)
      
      -- calculate boxes from output regression
      local ex_boxes = bbox_from_regression(anchors[i]:cuda(), y_reg:cuda(), val_mean, val_stds)
      
      -- only positive boxes (label > 0) and theirs corresponding gt 
      local pos_ex_boxes = ex_boxes[torch.eq(pos_boxes[i], 1)]
      pos_ex_boxes = pos_ex_boxes:reshape(pos_ex_boxes:size(1)/4, 4)

      table.insert(pos, pos_ex_boxes)
      
      --print(pos_ex_boxes:round())
      -- calc overlap of positive boxes and scaled rois
      local overlap = boxoverlap(pos_ex_boxes, scaled_rois[i])
      --print(pos_ex_boxes)
      -- calculate mean regression error and mean number of correct boxes (IoU > 0.7)
      local mean_overlap = 0
      local correct_boxes = 0
      
      for j = 1, pos_ex_boxes:size(1) do
        mean_overlap = mean_overlap + overlap[j][idx_ex_gt[i][j]]
        if (overlap[j][idx_ex_gt[i][j]] >= 0.7) then
          correct_boxes = correct_boxes + 1
        end
      end
      
      correct_boxes = correct_boxes / pos_ex_boxes:size(1)
      mean_overlap = mean_overlap / pos_ex_boxes:size(1)
      val_reg_accuracy = val_reg_accuracy + mean_overlap
      val_reg_correct = val_reg_correct + correct_boxes
      
      -- transform labels to batch
      local target_out_l = torch.Tensor(1, target_labels:size(1), target_labels:size(2)):cuda()
      target_out_l[{1,{},{}}] = target_labels
      
      local y_labels = y[1]:permute(2,3,1):cuda()
      y_labels = y_labels:reshape(y_labels:size(1)*y_labels:size(2), y_labels:size(3) / 3, 3)
      
      -- batch x classes x (height x widht) x boxes
      local y_out_l = torch.Tensor(1, y_labels:size(1), y_labels:size(2), y_labels:size(3)):cuda() 
      y_out_l[{1,{},{},{}}] = y_labels
      y_out_l = y_out_l:permute(1,4,2,3)
      
      -- only positive target boxes
      y_reg[torch.ne(target_bbox_loss,1)] = 0
      
      local out = {}
      
      -- only use positive and negative anchors for loss function
      table.insert(out, y_out_l:contiguous())
      table.insert(out, y_reg)
      
      target = {}
      table.insert(target, target_out_l:contiguous())
      table.insert(target, target_reg)
      

      -- calculate loss
      local err = criterion:forward(out, target)
      val_loss = val_loss + err
      
      -- calculate fg and bg accuracy
      local acc_fg, acc_bg = check_error(target_labels, target_weights, y_out_l)
      val_fg_acc = val_fg_acc + acc_fg
      val_bg_acc = val_bg_acc + acc_bg
      
  end 
  val_reg_accuracy = val_reg_accuracy / #val_batch[1]
  val_reg_correct = val_reg_correct / #val_batch[1]
  val_fg_acc = val_fg_acc / #val_batch[1]
  val_bg_acc = val_bg_acc / #val_batch[1]
  val_loss = val_loss / #val_batch[1]
  
    print(string.format('Epoch: [%d][VALIDATION SUMMARY] Total Time(s): %.2f  loss:%.4f   acc_fg:%.4f   acc_bg:%.4f   acc_reg:%.4f corr_reg:%.4f' , epoch, torch.toc(tic), val_loss, val_fg_acc, val_bg_acc, val_reg_accuracy, val_reg_correct ))
  print('\n')
  
  
  for i = 1,imgCount do
      --calculate back to original image (bgr->bgr and mean/std calculation)
      local im = torch.ByteTensor(val_batch[1][perm[i]]:size())
      -- change back from bgr to rgb
      im[1][{{},{}}] = val_batch[1][perm[i]][1][{{},{}}]
      im[2][{{},{}}] = val_batch[1][perm[i]][2][{{},{}}]
      im[3][{{},{}}] = val_batch[1][perm[i]][3][{{},{}}]
      
      im = im:index(1, torch.LongTensor{3,2,1})
          
      -- add mean to image
      im = img_from_mean(im, conf.image_means)
         
      local pos_ex_boxes = pos[perm[i]]
      temp = pos[perm[i]]
      local im_size = torch.Tensor{im:size(2), im:size(3)}

      gt = scaled_rois[perm[i]]
      -- draw all gt boxes into image
      for j = 1,gt:size(1) do       
        im = image.drawRect(im, gt[{j,2}], gt[{j,1}], gt[{j,4}], gt[{j,3}], {lineWidth = 1, color = {0, 255, 0}})    
      end
      
      image.save(conf.save_model_state.. 'Images/valGt' .. i .. '.png', im)
      
      -- draw all positive boxes into image
      for j = 1,pos_ex_boxes:size(1) do
        local x2, y2 = 0
        local col = torch.Tensor(3)
        col[1] = torch.random(1,255)
        col[2] = torch.random(1,255)
        col[3] = torch.random(1,255)
        if(pos_ex_boxes[{j,1}] < im_size[1] and pos_ex_boxes[{j,2}] < im_size[2] and pos_ex_boxes[{j,1}] > 0 and pos_ex_boxes[{j,2}] > 0) then
          if (pos_ex_boxes[{j,3}] > im:size(2)) then
            x2 = im_size[1]
          else
            x2 = pos_ex_boxes[{j,3}]
          end
          
          if (pos_ex_boxes[{j,4}] > im:size(3)) then
            y2 = im_size[2]
          else
            y2 = pos_ex_boxes[{j,4}]
          end
          im = image.drawRect(im, pos_ex_boxes[{j,2}], pos_ex_boxes[{j,1}],pos_ex_boxes[{j,4}], pos_ex_boxes[{j,3}], {lineWidth = 1, color = col})    
          --draw_rect(im, pos_ex_boxes[{j,1}], pos_ex_boxes[{j,2}], x2, y2, {255, 0, 0}) 
        end
      end
            
      image.save(conf.save_model_state.. 'Images/valEx' .. i .. '.png', im)
      
    end
end
