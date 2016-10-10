
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
local list_labels = {}
batch_roidbs ={}

local function paramsForEpoch(epoch)
    if conf.learningRate ~= 0.0 then -- if manually specified
        return { }
    end
    local regimes = {
        -- start, end,    LR,   WD,
        {  1,    1,   1e-4,   5e-4, },
        {  2,    12,   1e-3,   5e-4, },
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

function train()
  model:training()
  epoch = epoch or 1

  train_loss = 0
  train_loss_reg = 0
  train_loss_cls = 0
  
  batchNumber = 0

  
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
  train_loss_reg = train_loss_reg / epochL
  train_loss_cls = train_loss_cls / epochL

  
  
  print(string.format('Epoch: [%d][TRAINING SUMMARY] Total Time(s): %.2f   loss:%.4f   loss_cls:%.4f   loss_reg:%.4f', epoch, torch.toc(tic), train_loss, train_loss_reg, train_loss_cls))
  print('\n')

  collectgarbage()

  if epoch % conf.save_epoch == 0 then
    model:clearState()
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



--local gradient = torch.CudaTensor()
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
    local loss_cls = 0
    local loss_reg = 0
    
    
    feval = function(x)
      collectgarbage()
                  
      if x ~= parameters then
        parameters:copy(x)
      end
      
      gradParameters:zero()
      
      local mini_batch = {}
      local target_output = {}
      
      table.insert(mini_batch, train_batch[1]:cuda())
      table.insert(mini_batch, train_batch[2]:cuda())
      
      local target_label = train_batch[4]
      --target_label:resize(1, 1, train_batch[4]:size(1)):copy(train_batch[4])
      table.insert(target_output, target_label)
      
      table.insert(target_output, train_batch[5]:cuda())
      
      output = model:forward(mini_batch)
            
      local size = output[2]:size()
      local nr_rois = output[1]:size(1)
      output[2][train_batch[6]:ne(1)] = 0
      output[2] = output[2]:reshape(size)

            
      collectgarbage()
      loss_cls = loss_cls + conf.weight_l1crit * sl1:forward(output[2], target_output[2]) / nr_rois
      loss_reg = loss_reg + conf.weight_scec * log:forward(output[1], target_output[1]) /nr_rois
      f = loss_cls + loss_reg
      
      local gradient = {}
      table.insert(gradient, log:backward(output[1], target_output[1]) * conf.weight_scec)
      table.insert(gradient, sl1:backward(output[2], target_output[2]) * conf.weight_l1crit)
      
      if conf.divGrad == 1 then
        gradient[1]:div(nr_rois)
        gradient[2]:div(nr_rois)
      end
      
      
      model:backward(mini_batch, gradient)   
      
      return f, gradParameters
  end
            
  optim.sgd(feval, parameters, optimState)   
  
  --local para, grad = model:parameters()
  collectgarbage()
  
  assert(parameters:storage() == model:parameters()[1]:storage())

  
  if model.needsSync then
    model:syncParameters()
  end
  
  cutorch.synchronize()
  
  batchNumber = batchNumber + 1
  
  train_loss = train_loss + f 
  train_loss_reg = train_loss_reg + loss_reg
  train_loss_cls = train_loss_cls + loss_cls

  
  print(('Epoch: [%d][%d/%d]\tTime(s) %.3f  loss %.4f  LR %.0e loss_reg: %.4f loss_cls: %.4f '):format(
      epoch, batchNumber, math.floor(numOfTrainImages/conf.batch_size), timer:time().real, f, 
      optimState.learningRate, loss_reg, loss_cls))
 
end
