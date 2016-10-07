
if (conf.load_old_network ~= true) then
  optimState = {
    learningRate = conf.learningRate,    
    learningRateDecay = 0.0,
    weightDecay = conf.weightDecay,
    momentum = conf.momentum, 
    dampening = 0.0
  }
end



parameters,gradParameters = model:getParameters()

local batchNumber
local numOfTrainImages
local processedImages
local indices = torch.CudaTensor()
local gt_boxes = {}
local pos_boxes ={}
local reg_for_boxes = {}
local save_images = {}
local firstImages = true
local im_sizes = {}
batch_roidbs ={}

local function paramsForEpoch(epoch)
    if conf.learningRate ~= 0.0 then -- if manually specified
        return { }
    end
    local regimes = {
        -- start, end,    LR,   WD,
        {  1,    12,   1e-3,   5e-4, },
        {  13,    20,   1e-4,   5e-4, },
        { 19,     1e8,   1e-4,   5e-4  },
        { 13,     1e8,   1e-4,   0 },
        { 21,     1e8,   5e-5,   0 },
    }

    for _, row in ipairs(regimes) do
        if epoch >= row[1] and epoch <= row[2] then
            return { learningRate=row[3], weightDecay=row[4], shedule=regimes }, epoch == row[1]
        end
    end
end

function check_error( labels, targets) 
    local _ , ex_class = torch.max(labels,2)
    ex_class = ex_class:cuda()
    local target = targets:cuda()
    local pos = torch.sum(ex_class:eq(target)) / target:size(1)
    local nr_pos = torch.sum(target:lt(21))
    local true_pos = 0
    local nr_neg = torch.sum(target:eq(21))
    local false_neg = 0
    
    if nr_pos > 0 then
      true_pos = torch.sum(ex_class[target:lt(21)]:eq(target[target:lt(21)])) / nr_pos
      false_pos = torch.sum(ex_class[target:eq(21)]:lt(21)) / nr_neg
    end
    
    local false_pos = torch.sum(ex_class[target:eq(21)]:lt(21)) / nr_neg
    return pos, false_neg, false_pos, true_pos
end


function train()
  model:training()
  epoch = epoch or 1

  train_loss = 0
  train_reg_accuracy = 0
  train_reg_correct = 0
  
  train_corr = 0
  train_true_pos = 0
  train_false_neg = 0
  train_false_pos = 0
  
  batchNumber = 0
  processedImages = 0
  firstImages = true
  
  gt_boxes = {}
  pos_boxes ={}
  reg_for_boxes = {}
  save_images = {}
  im_sizes = {}
  unscaled_gt = {}
  
  if conf.learningRate == 0 then
     local params, newRegime = paramsForEpoch(epoch)
     if newRegime then
        optimState.learningRate = params.learningRate
        optimState.weightDecay = params.weightDecay
     end
     
     learning_rate_shedule = params.shedule
   end

  print(color.blue '==>'.." online epoch # " .. epoch .. ' [batchSize = ' .. conf.batch_size .. ']')
  
  local tic = torch.tic()
  
  indices = torch.randperm(#image_roidb_train):long():split(conf.batch_size)
  
  -- remove last element if not full batch_size so that all the batches have equal size
  if (#image_roidb_train % conf.batch_size ~= 0) then
    indices[#indices] = nil
  end
  
  numOfTrainImages = #indices * conf.batch_size
  
  epochL = #indices

  cutorch.synchronize()
  
  for t,v in ipairs(indices) do
    if torch.sum(parameters:ne(parameters)) > 0 then
      return
    end
    batch_roidbs = generate_batch_roidbs(v, image_roidb_train)    
    train_batch(batch_roidbs)
    
    if t == epochL then
      break
    end
  end
  
  cutorch.synchronize()

  train_loss = train_loss / epochL
  train_reg_accuracy = train_reg_accuracy/epochL
  train_reg_correct = train_reg_correct/epochL
  train_corr = train_corr /epochL
  train_true_pos = train_true_pos / epochL
  train_false_neg = train_false_neg / epochL
  train_false_pos = train_false_pos / epochL
  
  
  print(string.format('Epoch: [%d][TRAINING SUMMARY] Total Time(s): %.2f   loss:%.4f   reg_acc:%.4f   reg_corr:%.4f', epoch, torch.toc(tic), train_loss, train_reg_accuracy, train_reg_correct))
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
local target_label = torch.CudaTensor()


function train_batch(roidbs)

    cutorch.synchronize()
    collectgarbage()
    timer:reset()
    
    local train_batch = fast_rcnn_generate_minibatch(roidbs)
      
    if(train_batch[4]:dim() == 0) then
      return
    end
    
    local f = 0
    local reg_acc = 0
    local reg_correct = 0
    local imgCount = 12
    
    local mini_batch = {}
    local target_ouput = {}
    
    feval = function(x)
      collectgarbage()
                  
      if x ~= parameters then
        parameters:copy(x)
      end
      
      gradParameters:zero()
      
      table.insert(mini_batch, train_batch[1]:cuda())
      table.insert(mini_batch, train_batch[2]:cuda())
      
      
      target_label:resize(1, 1, train_batch[4]:size(1)):copy(train_batch[4])
      table.insert(target_ouput, target_label)
      
      table.insert(target_ouput, train_batch[5]:cuda())
      
      local sm = nn.LogSoftMax():cuda()
      output = model:forward(mini_batch)
            
      local size = output[2]:size()
      local nr_rois = output[1]:size(1)
      output[2][train_batch[6]:ne(1)] = 0
      output[2]:resize(size)


      output[1] = output[1]:permute(2,1)
      output[1]:resize(1, output[1]:size(1), 1, output[1]:size(2))

      local corr, false_neg, false_pos, true_pos = check_error(output[1], train_batch[4])

      train_corr = train_corr + corr
      train_true_pos = train_true_pos + true_pos
      train_false_neg = train_false_neg + false_neg
      train_false_pos = train_false_pos + false_pos
      
      print (('Corr: %.2f  FN: %.2f  FP: %.2f  TP: %.2f'):format(corr, false_neg, false_pos, true_pos))
      f = criterion:forward(output, target_ouput)
      
      if f > 100 then
        print(output[1][{1,{}, 1,1}])
        print(output[1][{1,{}, 1,2}])
        print(target_label[{1,1,1}])
        print(target_label[{1,1,2}])
      end
      
      
      

      gradient = criterion:backward(output, target_ouput)

      -- restore original dimensionality
      gradient[1] = gradient[1]:permute(1,3,4,2)
      gradient[1]:resize(train_batch[3]:size(1), conf.numClasses+1)
      
      
      if conf.divGrad == 1 then
        gradient[1]:div(nr_rois)
        gradient[2]:div(nr_rois)
      end
      
      
      model:backward(mini_batch, gradient)      
      --------------------------------------------------------------------------------------
      -- Calculate the IoU error

      -- get the regression of positive rois
      if torch.sum(train_batch[6]:eq(1)) > 0 then
        local reg = output[2][train_batch[6]:eq(1)] 
        reg:resize(reg:size(1)/4, 4)
        local temp = train_batch[5][train_batch[6]:eq(1)]:resize(reg:size(1)/4,4)
        local idx = torch.lt(train_batch[4], 21):double()
        
        -- get the gt indexes for positive rois
        local gt_idx = train_batch[7][torch.eq(idx, 1)]
        
        -- get the rois of positive rois
        idx = idx:cat(idx,2):cat(idx,2):cat(idx, 2)
        local rois = train_batch[3][torch.eq(idx,1)]
        
        local new_rois = torch.Tensor()
        
        if rois:dim() > 0 then
          rois:resize(rois:size(1)/4, 4)  
              
          new_rois = bbox_from_regression(rois, reg, mean_boxes[2], stds_boxes[2]:reshape(1,4))
          -- restrict rois to image size
          local size_im = torch.Tensor{train_batch[1][1]:size(2), train_batch[1][1]:size(3)}
          new_rois = restrict_rois_to_image_size(new_rois, size_im)
          new_rois = torch.round(new_rois)

          -- calculate overlap between new_rois and the scaled gt_boxes
          local overlap = boxoverlap(new_rois:float(), train_batch[8]:float())
          
          for j = 1, new_rois:size(1) do
            reg_acc = reg_acc + overlap[j][gt_idx[j]]
            --print(new_rois[j]:cat(train_batch[8][gt_idx[j]]):cat(torch.CudaTensor({overlap[j][gt_idx[j]]})))
            if overlap[j][gt_idx[j]] > 0.5 then
              reg_correct = reg_correct + 1
            end
          end
          
          local idx_img = train_batch[2][{{},1}][torch.ne(train_batch[4], 21)]
          
          
          reg_correct = reg_correct / rois:size(1)    
          reg_acc = reg_acc / rois:size(1)
        end
      
        local idx_img = train_batch[2][{{},1}][torch.lt(train_batch[4], 21)]:double()
        idx_img = idx_img:cat(idx_img, 2):cat(idx_img, 2):cat(idx_img, 2)
        
        local gt_boxes_idx = 1
        for i = 1, #roidbs do
          if(processedImages + i) <= imgCount then
            table.insert(save_images, train_batch[1][i])
            local image_rois = new_rois[torch.eq(idx_img, i)]
            if image_rois:dim() > 0 then
              image_rois:resize(image_rois:size(1)/4, 4)
            end
            table.insert(pos_boxes, image_rois)
            table.insert(gt_boxes, train_batch[8][{{gt_boxes_idx, gt_boxes_idx + roidbs[i].gt_boxes:size(1) - 1}, {}}])
            table.insert(im_sizes, train_batch[9][i])          
            gt_boxes_idx = gt_boxes_idx + roidbs[i].gt_boxes:size(1)
          else 
            break
          end  
          
        end
        
        processedImages = processedImages + conf.batch_size
      end
      
      f = f / conf.batch_size
      
      return f, gradParameters--:div(opt.batchSize) --------------------------------------------------
  end
            
  optim.sgd(feval, parameters, optimState)   
  
  --local para, grad = model:parameters()
  
  assert(parameters:storage() == model:parameters()[1]:storage())

  
  if model.needsSync then
    model:syncParameters()
  end
  
  cutorch.synchronize()
  
  batchNumber = batchNumber + 1
  
  train_loss = train_loss + f 
  train_reg_accuracy = train_reg_accuracy + reg_acc
  train_reg_correct = train_reg_correct + reg_correct
  
  print(('Epoch: [%d][%d/%d]\tTime(s) %.3f  loss %.4f  LR %.0e  RA:%.4f  RC:%.4f'):format(
      epoch, batchNumber, math.floor(numOfTrainImages/conf.batch_size), timer:time().real, f, 
      optimState.learningRate, reg_acc, reg_correct ))
  
  if processedImages >= imgCount and firstImages then
    for i = 1,imgCount do
      --calculate back to original image (bgr->bgr and mean/std calculation)

      local im_size = im_sizes[i]
      
      save_images[i] = save_images[i][{{},{1, im_size[1]}, {1, im_size[2]}}]
      
      -- change back from brg to rgb
      save_images[i] = save_images[i]:index(1, torch.LongTensor{3,2,1})
         
      -- add mean to image
      save_images[i] = img_from_mean(save_images[i], conf.image_means)
      
      local gt = gt_boxes[i]
      local pos_ex_boxes = pos_boxes[i]
      
      for j = 1,gt:size(1) do
         save_images[i] = image.drawRect( save_images[i]:byte(), gt[{j,2}], gt[{j,1}], gt[{j,4}], gt[{j,3}], {lineWidth = 1, color = {0, 255, 0}})    
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

            if (pos_ex_boxes[{j,3}] > im_size[1]) then
              x2 = im_size[1]
            else
              x2 = pos_ex_boxes[{j,3}]
            end
            
            if pos_ex_boxes[{j,4}] > im_size[2] then
              y2 = im_size[2]
            else
              y2 = pos_ex_boxes[{j,4}]
            end
            save_images[i] = image.drawRect(save_images[i], pos_ex_boxes[{j,2}], pos_ex_boxes[{j,1}],pos_ex_boxes[{j,4}], pos_ex_boxes[{j,3}], {lineWidth = 1, color = col})    
          end
        end
      end
            
      image.save(conf.save_model_state.. 'Images/trainEx' .. i .. '.png', save_images[i])
      
    end
    firstImages = false
  end
end
