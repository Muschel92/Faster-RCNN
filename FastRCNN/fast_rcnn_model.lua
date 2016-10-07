

if (conf.load_old_network) then
  model = torch.load(conf.load_trainState_path .. conf.model_type)
  trainState = torch.load(conf.load_trainState_path .. conf.trainState_name )
  optimState = torch.load(conf.load_trainState_path .. conf.optimState_name )
elseif (conf.create_network) then
  print '==> executing all'
  dofile('FastRCNN/fast_rcnn_create_model.lua')
else
  print '==> load model'
  model = torch.load(conf.network_path .. conf.model_type)
  torch.manualSeed(conf.rng_seed)
end

model:cuda()

model = makeDataParallel(model, opt.nGPU) 

criterion = nn.ParallelCriterion():cuda()
--weights = torch.ones(conf.numClasses +1) *conf.fg_weights
--weights[conf.numClasses +1] = conf.bg_weights
log = cudnn.SpatialCrossEntropyCriterion():cuda()

--sl1 = nn.MSECriterion():cuda()
--sl1 = nn.AbsCriterion():cuda()
sl1 = nn.SmoothL1Criterion():cuda()
if conf.sizeAverage_sl1 == 0 then
  sl1.sizeAverage =false
end
if conf.sizeAverage_log == 0 then
  log.sizeAverage= false
end
--criterion:add(log, conf.weight_scec):add(sl1, conf.weight_l1crit)
