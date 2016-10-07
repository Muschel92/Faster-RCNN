
accFgLogger = optim.Logger(paths.concat(conf.save_model_state , 'accuracyFg.log'))
accFgLogger:setNames{'% mean accuracy Fg (train set)', '% mean accuracy Fg (val set)'}
accFgLogger:style{'+-', '+-'}
accFgLogger.showPlot = false

accBgLogger = optim.Logger(paths.concat(conf.save_model_state , 'accuracyBg.log'))
accBgLogger:setNames{'% mean accuracy Bg (train set)', '% mean accuracy Bg(val set)'}
accBgLogger:style{'+-', '+-'}
accBgLogger.showPlot = false

lossLogger = optim.Logger(paths.concat(conf.save_model_state, 'loss.log'))
lossLogger:setNames{'% mean loss (train set)', '% mean loss (val set)'}
lossLogger:style{'+-', '+-'}
lossLogger.showPlot = false

reg_accuracyLogger = optim.Logger(paths.concat(conf.save_model_state, 'reg_accuracy.log'))
reg_accuracyLogger:setNames{'% regression accuracy (train set)', '% regression accuracy (val set)'}
reg_accuracyLogger:style{'+-', '+-'}
reg_accuracyLogger.showPlot = false

reg_correctLogger = optim.Logger(paths.concat(conf.save_model_state, 'reg_correct.log'))
reg_correctLogger:setNames{'% regression correct boxes accuracy (train set)', '% regression correct boxes accuracy (val set)'}
reg_correctLogger:style{'+-', '+-'}
reg_correctLogger.showPlot = false

train_fg_acc = 0
train_bg_acc = 0 
train_loss = 0
train_reg_accuracy = 0
train_reg_correct = 0
train_ntrl_acc = 0

val_fg_acc = 0
val_bg_acc = 0 
val_reg_accuracy = 0
val_reg_correct = 0
val_loss = 0

learning_rate_shedule = {}

local acc_fg_dir = paths.concat(conf.save_model_state, 'accuracyFg.png')
local acc_bg_dir = paths.concat(conf.save_model_state, 'accuracyBg.png')
local loss_dir = paths.concat(conf.save_model_state, 'loss.png')      
local reg_accuracy_dir = paths.concat(conf.save_model_state, 'reg_accuracy.png')
local reg_correct_dir = paths.concat(conf.save_model_state, 'reg_correct.png') 

function logging()
    
    accFgLogger:add{train_fg_acc, val_fg_acc}
    accFgLogger:plot()
    accBgLogger:add{train_bg_acc, val_bg_acc}
    accBgLogger:plot()
    lossLogger:add{train_loss, val_loss}
    lossLogger:plot()
    reg_accuracyLogger:add{train_reg_accuracy, val_reg_accuracy}
    reg_accuracyLogger:plot()
    reg_correctLogger:add{train_reg_correct, val_reg_correct}
    reg_correctLogger:plot()
    
end


function writeReport()

  os.execute(('convert -density 200 %saccuracyFg.log.eps %saccuracyFg.png'):format(conf.save_model_state,conf.save_model_state))
  os.execute(('convert -density 200 %saccuracyBg.log.eps %saccuracyBg.png'):format(conf.save_model_state,conf.save_model_state))
  os.execute(('convert -density 200 %sloss.log.eps %sloss.png'):format(conf.save_model_state,conf.save_model_state))
  os.execute(('convert -density 200 %sreg_accuracy.log.eps %sreg_accuracy.png'):format(conf.save_model_state,conf.save_model_state))
  os.execute(('convert -density 200 %sreg_correct.log.eps %sreg_correct.png'):format(conf.save_model_state,conf.save_model_state))

  local file = io.open(conf.save_model_state..'report.html','w')
  file:write(([[
      <!DOCTYPE html>
      <html>
      <body>
      <h4>Log: %s</h4>
      <h4>Epoch: %s</h4>
      <h4> Accuracy_Fg: </h4>
      <img src=%s> </br>
      <h4> Accuracy_Bg: </h4>
      <img src=%s> </br>
      <h4> Loss: </h4>
      <img src=%s>
      <h4> Accuracy regression: </h4>
      <img src=%s>
      <h4> Accuracy regression correct: </h4>
      <img src=%s>
      <h4> Accuracies in Val-Set:</h4>
      <table>
      ]]):format(conf.save_model_state,epoch, 'accuracyFg.png', 'accuracyBg.png', 'loss.png', 'reg_accuracy.png', 'reg_correct.png' ))      
      -- ]]):format(conf.save_model_state,epoch,acc_fg_dir,acc_bg_dir, loss_dir, reg_accuracy_dir,reg_correct_dir ))      

  file:write('<tr> <td>'..'Accuraccy_foreground'..'</td> <td>'.. val_fg_acc ..'</td> </tr> \n')
  file:write('<tr> <td>'..'Accuraccy_background'..'</td> <td>'.. val_bg_acc ..'</td> </tr> \n')
  file:write('<tr> <td>'..'Regression_accuracy'..'</td> <td>'.. val_reg_accuracy ..'</td> </tr> \n')
  file:write('<tr> <td>'..'Regression correct'..'</td> <td>'.. val_reg_correct ..'</td> </tr> \n')

  mean_acc = val_bg_acc + val_fg_acc /2
  file:write('<tr> <td> MEAN </td> <td>'.. mean_acc ..'</td> </tr> \n')

-----------------------------------------------------------------------------------------

  file:write([[</table>
  <h4> OptimState: </h4>
  <table>
  ]])

  for k,v in pairs(optimState) do
    if torch.type(v) == 'number' then
      file:write('<tr> <td>'..k..'</td> <td>'..v..'</td> </tr> \n')
    end
  end

-----------------------------------------------------------------------------------------

  file:write([[</table>
  <h4> Opts: </h4>
  <table>
  ]])

  for k,v in pairs(conf) do
    if torch.type(v) == 'number' or torch.type(v) == 'string' then
      file:write('<tr> <td>'..k..'</td> <td>'..v..'</td> </tr> \n')
    end
  end

-----------------------------------------------------------------------------------------
  if conf.learningRate == 0.0 then
	  file:write([[</table>
	  <h4> Learning Rate Shedule: </h4>
	  <table>
	  ]])
	
	  for k,v in pairs(learning_rate_shedule) do
	    if k == 1 then
		file:write('<tr> <td>'..'Begin epoch' ..'</td> <td>'..'End epoch' ..'</td><td>'..'learningRate' ..'</td><td>'..'WeightDecy' ..'</td> </tr> \n')
	    end
	    file:write('<tr> <td>'..v[1]..'</td> <td>'..v[2]..'</td> <td>'..v[3]..'</td> <td>'..v[4]..'</td> </tr> \n')
	  end
  end
-----------------------------------------------------------------------------------------

  file:write([[
    </table>
    <h4> Train Images </h4>
    gt image  - gt und prediction image </br>
    ]])

--input and output images
  local imgCount = 12
  for i = 1, imgCount do
    input_dir = paths.concat(conf.save_model_state , 'Images/trainGt' .. i .. '.png')      
    label_dir = paths.concat(conf.save_model_state , 'Images/trainEx' .. i .. '.png')      

    file:write(([[
      <h5> ImagePair: %s </h5>    
      <img src=%s>
      <img src=%s>
      </br>
      ]]):format(i, input_dir, label_dir))     
  end

-----------------------------------------------------------------------------------------
  if (conf.do_validation) then
    file:write([[
      </table>
      <h4> Val Images </h4>
      input image  - output image - correct label - correct predicted </br>
      ]])

  --input and output images
    imgCount = 12
    for i = 1, imgCount do
      input_dir = 'Images/valGt' .. i .. '.png'     
      label_dir = 'Images/valEx' .. i .. '.png'    

      file:write(([[
        <h5> ImagePair: %s </h5>    
        <img src=%s>
        <img src=%s>
        </br>
        ]]):format(i, input_dir, label_dir))     
    end
  end
-----------------------------------------------------------------------------------------

-----------------------------------------------------------------------------------------
  file:write([[
      <h4> Model: </h4>
      <pre> ]])
    file:write(tostring(model))

    file:write('</pre></body></html>')
    file:close()



end
