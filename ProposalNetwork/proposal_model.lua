

if (conf.load_old_network) then
  model = torch.load(conf.load_trainState_path .. conf.model_type)
  trainState = torch.load(conf.load_trainState_path .. conf.trainState_name )
  optimState = torch.load(conf.load_trainState_path .. conf.optimState_name )
elseif (conf.create_network) then
  dofile('ProposalNetwork_create_model.lua')
else
  print(conf.network_path .. conf.model_type)
  model = torch.load(conf.network_path .. conf.model_type)
  --print(model)
  -- set random seed
  torch.manualSeed(conf.rng_seed)
end

model = model:cuda()

model = makeDataParallel(model, opt.nGPU) 

criterion = nn.ParallelCriterion():cuda()
log = cudnn.SpatialCrossEntropyCriterion(torch.Tensor{1, conf.bg_weights,0}):cuda()
sl1 = nn.SmoothL1Criterion():cuda()

if conf.sizeAverage_cls == 0 then
  log.sizeAverage = false
end

if conf.sizeAverage_reg == 0 then
  sl1.sizeAverage =false
end
criterion:add(log, conf.weight_scec):add(sl1, conf.weight_l1crit)
