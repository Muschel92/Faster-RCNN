local batchNumber = 0
local numOfTrainImages = 0
local processedImages = 0
local indices = torch.CudaTensor()
local gt_boxes = {}
local pos_boxes ={}
local reg_for_boxes = {}
local save_images = {}
batch_roidbs ={}


function validation()
  model:evaluate()
  epoch = epoch or 1

  val_loss = 0
  val_reg_accuracy = 0
  val_reg_correct = 0
  
  processedImages = 0
  
  gt_boxes = {}
  pos_boxes ={}
  reg_for_boxes = {}
  save_images = {}
    
  local tic = torch.tic()
  
  indices = torch.randperm(#image_roidb_val):long():split(conf.batch_size)
  
  -- remove last element if not full batch_size so that all the batches have equal size
  if (#image_roidb_val % conf.batch_size ~= 0) then
    indices[#indices] = nil
  end
  
  numOfTrainImages = #indices * conf.batch_size


  cutorch.synchronize()
  
  for t,v in ipairs(indices) do
    batch_roidbs = generate_batch_roidbs(v, image_roidb_val)  
    
    val_batch(batch_roidbs)
  end
  
  cutorch.synchronize()

  val_loss = val_loss / #indices
  val_reg_accuracy = val_reg_accuracy/ #indices
  val_reg_correct = val_reg_correct/ #indices
  
  
  print(string.format('Epoch: [%d][VALIDATION SUMMARY] Total Time(s): %.2f   loss:%.4f   reg_acc:%.4f   reg_corr:%.4f', epoch, torch.toc(tic), val_loss, val_reg_accuracy, val_reg_correct))
  print('\n')

  collectgarbage()

  if epoch % conf.save_epoch == 0 then
    if torch.type(model) == 'nn.DataParallelTable' then
      torch.save(paths.concat(conf.save_model_state, 'Models/model_' .. epoch .. '.t7'), model:get(1))
    else
      torch.save(paths.concat(conf.save_model_state, 'Models/model_' .. epoch .. '.t7'), model)
    end
    
    torch.save(paths.concat(conf.save_model_state, 'Models/optimState_' .. epoch .. '.t7'), optimState)
    local trainState = {}
    trainState.epoch = epoch
  end
  
end



local gradient = torch.CudaTensor()
local timer = torch.Timer()
local dataTimer = torch.Timer()




function val_batch(roidbs)

    cutorch.synchronize()
    collectgarbage()
    timer:reset()
    
    local train_batch = fast_rcnn_generate_minibatch(roidbs)
              
    local f = 0
    local reg_acc = 0
    local reg_correct = 0
    local imgCount = 0
    
    local mini_batch = {}
    local target_ouput = {}
    
    
    if (batchNumber == 0) then
      imgCount = 12 
      --conf.batch_size < 16 and conf.batch_size or 16
    end
    
    feval = function(x)
      collectgarbage()
                  
      if x ~= parameters then
        parameters:copy(x)
      end
      
      gradParameters:zero()
      
      table.insert(mini_batch, train_batch[1]:cuda())
      table.insert(mini_batch, train_batch[2]:cuda())
      
      local target_label = torch.CudaTensor(1, 1, train_batch[4]:size(1)):copy(train_batch[4])
      table.insert(target_ouput, target_label)
      
      table.insert(target_ouput, train_batch[5]:cuda())
      
      output = model:forward(mini_batch)
      
      output[2][train_batch[6]:ne(1)] = 0
      local batch_label = torch.CudaTensor(1, output[1]:size(2), 1, output[1]:size(1)):copy(output[1]:permute(2,1))
      --batch_label = batch_label:permute(1,4,2,3)
      output[1] = batch_label

      f = criterion:forward(output, target_ouput)
      
      --------------------------------------------------------------------------------------
      -- Calculate the IoU error
      
      -- get the regression of positive rois
      local reg = output[2][train_batch[6]:eq(1)] 
      reg = reg:reshape(reg:size(1)/4, 4)
      local idx = torch.ne(train_batch[4], 21):double()
      
      -- get the gt indexes for positive rois
      local gt_idx = train_batch[7][torch.eq(idx, 1)]
      
      -- get the rois of positive rois
      idx = idx:cat(idx,2):cat(idx,2):cat(idx, 2)
      local rois = train_batch[3][torch.eq(idx,1)]
      
      local new_rois = torch.Tensor()
      
      if rois:dim() > 0 then
        rois = rois:reshape(rois:size(1)/4, 4)        
        reg = reg[torch.eq(idx,1)]
        reg = reg:reshape(reg:size(1)/4,4)
            
      -- calculate the new rois after regression
        new_rois = bbox_from_regression(rois, reg, mean_boxes[2], stds_boxes[2]:reshape(1,4))
        
        -- calculate overlap between new_rois and the scaled gt_boxes
        local overlap = boxoverlap(new_rois, train_batch[8])
        local temp_rec = 0
        local temp_acc = 0
        
        for j = 1, new_rois:size(1) do
          reg_correct = reg_correct + overlap[j][gt_idx[j]]
          if overlap[j][gt_idx[j]] > 0.5 then
            reg_acc = reg_acc + 1
          end
        end
        
        local idx_img = train_batch[2][{{},1}][torch.ne(train_batch[4], 21)]
        
        
        reg_correct = reg_correct / rois:size(1)    
        reg_acc = reg_acc / rois:size(1)
      end
    
      local idx_img = train_batch[2][{{},1}][torch.ne(train_batch[4], 21)]
      idx_img = idx_img:cat(idx_img, 2):cat(idx_img, 2):cat(idx_img, 2)

      
      for i = 1, #roidbs do
        if(processedImages + i) < imgCount then
          table.insert(save_images, train_batch[1][i])
          table.insert(gt_boxes, roidbs[i].gt_boxes)
          table.insert(pos_boxes, new_rois[torch.eq(idx_img, i)])
        end  
      end
      
      processedImages = processedImages + #roidbs
      
      f = f / conf.batch_size

      return f, gradParameters--:div(opt.batchSize) --------------------------------------------------
  end
            
  optim.sgd(feval, parameters, optimState)   
  
  --print('Different Parameters:')
  --print(parameters:eq(temp):size())
  
  assert(parameters:storage() == model:parameters()[1]:storage())
  
  if model.needsSync then
    model:syncParameters()
  end
  
  cutorch.synchronize()
  
  batchNumber = batchNumber + 1
  
  val_loss = val_loss + f
  val_reg_accuracy = val_reg_accuracy + reg_acc
  val_reg_correct = val_reg_correct + reg_correct
  
  if processedImages >= imgCount then
    for i = 1,imgCount do
      --calculate back to original image (bgr->bgr and mean/std calculation)
      
      -- change back from brg to rgb
      save_images[i]:index(1, torch.LongTensor{3,2,1})
         
      -- add mean to image
      save_images[i] = img_from_mean(save_images[i], conf.image_means)

      local im_size = torch.Tensor{ save_images[i]:size(2),  save_images[i]:size(3)}
      local gt = gt_boxes[i]
      local pos_ex_boxes = pos_boxes[i]

      --print(pos_ex_boxes:size())
      -- draw all gt boxes into image
      for j = 1,gt:size(1) do
         save_images[i] = image.drawRect( save_images[i], gt[{j,2}], gt[{j,1}], gt[{j,4}], gt[{j,3}], {lineWidth = 1, color = {0, 255, 0}})    
      end
      
      image.save(conf.save_model_state.. 'Images/trainGt' .. i .. '.png',  save_images[i])
      
      -- draw all positive boxes into image
      if pos_ex_boxes:dim() > 1 then
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
            save_images[i] = image.drawRect(save_images[i], pos_ex_boxes[{j,2}], pos_ex_boxes[{j,1}],pos_ex_boxes[{j,4}], pos_ex_boxes[{j,3}], {lineWidth = 1, color = col})    
            --draw_rect(im, pos_ex_boxes[{j,1}], pos_ex_boxes[{j,2}], x2, y2, {255, 0, 0}) 
          end
        end
      end
            
      image.save(conf.save_model_state.. 'Images/trainEx' .. i .. '.png', save_images[i])
      
    end
  end
end
