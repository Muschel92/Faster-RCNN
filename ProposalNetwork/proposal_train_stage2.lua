
if (conf.load_old_network ~= true) then
  optimState = {
    learningRate = conf.learningRate,    
    learningRateDecay = 0.0,
    weightDecay = conf.weightDecay,
    momentum = conf.momentum, 
    nesterov = true,
    dampening = 0.0
  }
end


parameters,gradParameters = model:getParameters()

local batchNumber = 0
local numOfTrainImages = 0
local processedImages = 0
local indices = torch.CudaTensor()
local gt_boxes = {}
local pos_boxes ={}
local reg_for_boxes = {}
local save_images = {}
local firstImages = true
batch_roidbs ={}


function check_error(labels, labels_weight, output)
    local y = output:permute(1,3,4,2)
    y:resize(9 * labels:size(2), 3)
    
    labels:resize(labels:size(2)*labels:size(3))    
    
    local _, t_l = torch.max(y[{{},{1,2}}], 2)
    
    local nr_fg = torch.sum(labels:eq(1))
    local nr_bg = torch.sum(labels:eq(2))
    local nr_neutral = torch.sum(labels:eq(3))
    
    local acc_fg = 0
    local acc_bg = 0
    local acc_fg_neutral = 0
    
    if nr_fg > 0 then
      acc_fg = torch.sum(t_l[labels:eq(1)]:eq(1)) / nr_fg
    end
    
    if nr_bg > 0 then
      acc_bg = torch.sum(t_l[labels:eq(2)]:eq(2)) / nr_bg
    end
    
    if nr_neutral > 0 then
      acc_fg_neutral = torch.sum(t_l[labels:eq(3)]:eq(1)) / nr_neutral
    end

  return acc_fg, acc_bg, acc_fg_neutral
end

local function paramsForEpoch(epoch)
    if conf.learningRate ~= 0.0 then -- if manually specified
        return { }
    end
    local regimes = {
        -- start, end,    LR,   WD,
        {  1,     12,   1e-3,   5e-4 },
        {13,     24,   1e-4,   5e-4  },
        { 17,     24,   1e-4,  5e-4 },
        { 25,     1e8,   1e-5,  5e-4 },
    }

    for _, row in ipairs(regimes) do
        if epoch >= row[1] and epoch <= row[2] then
            return { learningRate=row[3], weightDecay=row[4], shedule=regimes }, epoch == row[1]
        end
    end
end


function train()
  model:training()
  epoch = epoch or 1
  
  train_fg_acc = 0
  train_bg_acc = 0 
  train_loss = 0
  train_reg_accuracy = 0
  train_reg_correct = 0
  train_ntrl_acc = 0
  batchNumber = 0
  processedImages = 0
  gt_boxes = {}
  pos_boxes ={}
  reg_for_boxes = {}
  save_images = {}
  firstImages = true
  
  if conf.learningRate == 0 then
     local params, newRegime = paramsForEpoch(epoch)
     if newRegime then
        optimState.learningRate = params.learningRate
        optimState.weightDecay = params.weightDecay
     end
     
     learning_rate_shedule = params.shedule
   end
   
  
  --[[
  if epoch % conf.epoch_step == 0 then 
    optimState.learningRate = optimState.learningRate * 0.1
  end]]--

  print(color.blue '==>'.." online epoch # " .. epoch .. ' [batchSize = ' .. conf.batch_size .. ']')
  
  local tic = torch.tic()
  
  indices = torch.randperm(#image_roidb_train):long():split(conf.batch_size)
  
  -- remove last element if not full batch_size so that all the batches have equal size
  if (#image_roidb_train % conf.batch_size ~= 0) then
    indices[#indices] = nil
  end
  
  numOfTrainImages = #indices * conf.batch_size

  if #indices < conf.max_iter then
    epochL = #indices
  else
    epochL = conf.max_iter
  end

  cutorch.synchronize()

  
  for t,v in ipairs(indices) do
    -- queue jobs to data-workers
    --[[
    donkeys:addjob(
      -- the job callback (runs in data-worker thread)
      function() --load single batches
        local roidbs = generate_batch_roidbs(v, image_roidb_train)  
        local batch = proposal_generate_mini_batch(conf, roidbs)
        return batch
      end,
      -- the end callback (runs in the main thread)
      train_batch
    );
    ]]--
    
    batch_roidbs = generate_batch_roidbs(v, image_roidb_train)  
    
    train_batch(batch_roidbs)
    
    if t == epochL then
      break
    end
  end

  cutorch.synchronize()

  train_loss = train_loss / epochL
  train_fg_acc = train_fg_acc / epochL
  train_bg_acc = train_bg_acc / epochL
  train_reg_accuracy = train_reg_accuracy/epochL
  train_reg_correct = train_reg_correct/epochL
  train_ntrl_acc = 0

  
  

  --train_loss = loss_epoch
  --train_fg_acc = acc_fg_epoch
  --train_bg_acc = acc_bg_epoch
  
  print(string.format('Epoch: [%d][TRAINING SUMMARY] Total Time(s): %.2f   loss:%.4f   acc_fg:%.4f   acc_bg:%.4f acc_fg_nt: %.4f  reg_acc:%.4f   reg_corr:%.4f', epoch, torch.toc(tic), train_loss, train_fg_acc, train_bg_acc, train_ntrl_acc, train_reg_accuracy, train_reg_correct))
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




local mini_batch = torch.CudaTensor()
local target_labels = torch.CudaTensor()
local target_weights = torch.CudaTensor()
local target_reg = torch.CudaTensor()
local target_out_l = torch.CudaTensor()
local y_labels = torch.CudaTensor()
local y_out_l = torch.CudaTensor()
local output = torch.CudaTensor()
local y_reg = torch.CudaTensor()
local target_bbox_loss = torch.CudaTensor()

local timer = torch.Timer()
local dataTimer = torch.Timer()




function train_batch(roidbs)

    cutorch.synchronize()
    collectgarbage()
    timer:reset()
    
    local train_batch = proposal_generate_mini_batch(conf, roidbs)
              
    local f = 0
    local accuracy_fg = 0
    local accuracy_bg = 0
    local accuracy_fg_neurtral = 0
    local reg_acc = 0
    local reg_correct = 0
    local imgCount = 12
    
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

      mini_batch:resize(train_batch[1]:size()):copy(train_batch[1])
      output = model:forward(mini_batch)
      gradient = {}
      table.insert(gradient, torch.CudaTensor(output[1]:size()))
      table.insert(gradient, torch.CudaTensor(output[2]:size()))
      
      local n_cls = conf.rois_per_image
      local n_reg = 0
      
      for i = 1, #roidbs do
        -- get target
        target_labels:resize(train_batch[2][i]:size()):copy(train_batch[2][i])
        target_weights:resize(train_batch[3][i]:size()):copy(train_batch[3][i])
        target_reg:resize(train_batch[4][i]:size()):copy(train_batch[4][i])
        target_bbox_loss:resize(train_batch[5][i]:size()):copy(train_batch[5][i])
        
        n_reg = target_labels:size(1) * target_labels:size(2)
                     
        -- batch x (height x width) x boxes
        target_labels:resize(1, target_labels:size(1), target_labels:size(2))
        
        -- set neutral and negative bbox to zero
        target_reg[torch.ne(target_bbox_loss,1)] = 0
        
        -- transform output to target size
        --(heightx width) x boxes x 3 classes
        y_labels = output[1][i]:permute(2,3,1):cuda()
        y_labels:resize(1, y_labels:size(1)*y_labels:size(2), y_labels:size(3) / 3, 3)
        
        -- batch x classes x (height x widht) x boxes
        y_labels = y_labels:permute(1,4,2,3)
        
        -- number of anchor configurations ( ratios * scales)
        k = conf.total_anchors
        y_reg = output[2][i]:permute(2,3,1)
        y_reg:resize(y_reg:size(1)*y_reg:size(2)*k, 4)
        y_reg[torch.ne(target_bbox_loss,1)] = 0
        
        local im_size = torch.Tensor{mini_batch:size(3), mini_batch:size(4)}
      
        -- calculate anchors
        local ex_boxes = calculate_bbox_from_reg_output(roidbs[i], im_size, train_batch[6][i], y_reg, mean_boxes, stds_boxes)
        
        local pos_ex_boxes_idx = torch.gt(roidbs[i].bbox_targets[train_batch[6][i]][train_batch[7][i]][{{},1}], 0)
        local ex_gt_idx = roidbs[i].bbox_targets[train_batch[6][i]][train_batch[7][i]][{{},1}][torch.eq(pos_ex_boxes_idx,1)]
        pos_ex_boxes_idx = pos_ex_boxes_idx:cat(pos_ex_boxes_idx,2):cat(pos_ex_boxes_idx):cat(pos_ex_boxes_idx)
        
        --print(y_reg[torch.eq(target_bbox_loss,1)]:cat(target_reg[torch.eq(target_bbox_loss,1)], 2))      
        local pos_ex_boxes = ex_boxes[torch.eq(pos_ex_boxes_idx,1)]
        pos_ex_boxes = torch.round(pos_ex_boxes:reshape(pos_ex_boxes:size(1)/4, 4))
        
        local gt = torch.Tensor(roidbs[i].boxes:size()):copy(roidbs[i].boxes)
        -- check if image is flipped
        if train_batch[7][i] == 2 then
          gt = flip_rois(gt, roidbs[i].size)
        end
                
        gt = scale_rois(gt, roidbs[i].size, im_size)


        
        if (processedImages <= imgCount) then
          table.insert(save_images,mini_batch[i]:clone():float())
          table.insert(gt_boxes, gt)
          table.insert(pos_boxes, pos_ex_boxes)
        end
        
        local overlap = boxoverlap(pos_ex_boxes:float(), gt)
        
      --print(pos_ex_boxes)
      -- calculate mean regression error and mean number of correct boxes (IoU > 0.7)
        local mean_overlap = 0
        local correct_boxes = 0
                
        for j = 1, pos_ex_boxes:size(1) do
          mean_overlap = mean_overlap + overlap[j][ex_gt_idx[j]]
          if (overlap[j][ex_gt_idx[j]] >= 0.7) then
            correct_boxes = correct_boxes + 1
          end
      end
      
      correct_boxes = correct_boxes / pos_ex_boxes:size(1)
      mean_overlap = mean_overlap / pos_ex_boxes:size(1)
      reg_acc = reg_acc + mean_overlap
      reg_correct = reg_correct + correct_boxes
        
        
        if(processedImages <= imgCount) then
          table.insert(reg_for_boxes, y_reg)
        end
                
        y = {}
        
        -- only use positive and negative anchors for loss function
        table.insert(y, y_labels:contiguous())
        table.insert(y, y_reg)
        
        target = {}
        table.insert(target, target_labels:contiguous())
        table.insert(target, target_reg)
        
        -- Calc/add error
        f = f + conf.weight_l1crit *sl1:forward(y[2], target[2])/n_reg + conf.weight_scec * log:forward(y[1], target[1]) / n_cls
        local err = criterion:forward(y, target)
        
        -- f = f* (i-1)/i+ err*(1/i)
        
        grad_list = criterion:backward(y, target)
        acc_fg, acc_bg, acc_fg_neutral = check_error(target_labels, target_weights, y_labels)
        
        accuracy_fg = accuracy_fg + acc_fg
        accuracy_bg = accuracy_bg + acc_bg
        accuracy_fg_neurtral = accuracy_fg_neurtral + acc_fg_neutral
        
        -- transform gradients back to correct dimension size 
               
        grad_list[1] = grad_list[1][1]:permute(2,3,1):cuda()
        grad_list[1]:resize(output[1][i]:size(2), output[1][i]:size(3), output[1][i]:size(1))
        grad_list[1] = grad_list[1]:permute(3,1,2)
        grad_list[1]:div(n_cls)
        grad_list[2]:resize(output[2][i]:size(2), output[2][i]:size(3), output[2][i]:size(1))
        grad_list[2] = grad_list[2]:permute(3,1,2)
        grad_list[2]:div(n_reg)
               
        gradient[1][i]:copy(grad_list[1])
        gradient[2][i]:copy(grad_list[2])
        --print(gradParameters[gradParameters:gt(0)]:size())
        processedImages = processedImages + 1
      end
    
      model:backward(mini_batch, gradient)   
      
      f = f/ #roidbs
      accuracy_fg = accuracy_fg / #roidbs
      accuracy_bg = accuracy_bg / #roidbs
      reg_acc = reg_acc / #roidbs
      reg_correct = reg_correct / #roidbs
      accuracy_fg_neurtral = accuracy_fg_neurtral / #roidbs

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
  
  train_loss = train_loss + f
  train_fg_acc = train_fg_acc + accuracy_fg
  train_bg_acc = train_bg_acc + accuracy_bg
  train_reg_accuracy = train_reg_accuracy + reg_acc
  train_reg_correct = train_reg_correct + reg_correct
  train_ntrl_acc = train_ntrl_acc + accuracy_fg_neurtral
  
  print(('Epoch: [%d][%d/%d]\tTime(s) %.3f  loss %.4f  LR %.0e AF:%.4f BF %.4f  AFN: %.4f RA:%.4f  RC:%.4f'):format(
      epoch, batchNumber, math.floor(numOfTrainImages/conf.batch_size), timer:time().real, f, 
      optimState.learningRate, accuracy_fg, accuracy_bg, accuracy_fg_neurtral, reg_acc, reg_correct ))

  if processedImages >= imgCount and firstImages then
    for i = 1,imgCount do
      --calculate back to original image (bgr->bgr and mean/std calculation)
      
      -- change back from brg to rgb
      save_images[i] = save_images[i]:index(1, torch.LongTensor{3,2,1})
         
      -- add mean to image
      save_images[i] = img_from_mean(save_images[i], conf.image_means)

      local im_size = torch.Tensor{ save_images[i]:size(2),  save_images[i]:size(3)}
      local gt = gt_boxes[i]
      local pos_ex_boxes = pos_boxes[i]
      
      --print(pos_ex_boxes:size())
      -- draw all gt boxes into image
      for j = 1,gt:size(1) do
         save_images[i] = image.drawRect( save_images[i]:byte(), gt[{j,2}], gt[{j,1}], gt[{j,4}], gt[{j,3}], {lineWidth = 1, color = {0, 255, 0}})    
      end     
      
      image.save(conf.save_model_state.. 'Images/trainGt' .. i .. '.png',  save_images[i])
      
      -- draw all positive boxes into image
      for j = 1,pos_ex_boxes:size(1) do
        local x2, y2 = 0
        local col = torch.Tensor(3)
        col[1] = torch.random(1,255)
        col[2] = torch.random(1,255)
        col[3] = torch.random(1,255)
        if(pos_ex_boxes[{j,1}] < im_size[1] and pos_ex_boxes[{j,2}] < im_size[2] and pos_ex_boxes[{j,1}] > 0 and pos_ex_boxes[{j,2}] > 0) then
          if (pos_ex_boxes[{j,3}] > im_size[2]) then
            x2 = im_size[1]
          else
            x2 = pos_ex_boxes[{j,3}]
          end
          
          if (pos_ex_boxes[{j,4}] > im_size[1]) then
            y2 = im_size[2]
          else
            y2 = pos_ex_boxes[{j,4}]
          end
          save_images[i] = image.drawRect(save_images[i]:byte(), pos_ex_boxes[{j,2}], pos_ex_boxes[{j,1}],pos_ex_boxes[{j,4}], pos_ex_boxes[{j,3}], {lineWidth = 1, color = col})    
          --draw_rect(im, pos_ex_boxes[{j,1}], pos_ex_boxes[{j,2}], x2, y2, {255, 0, 0}) 
        end
      end
            
      image.save(conf.save_model_state.. 'Images/trainEx' .. i .. '.png', save_images[i])
      
    end
    
    firstImages = false
  end
  
  
    
end
  
