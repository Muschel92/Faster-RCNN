require('torch')

eps = 1e-5

function bbox_transform (ex_boxes, gt_boxes)
    -- calculates width/ height and center point of ex_boxes
    local ex_width = ex_boxes[{{}, 3}] - ex_boxes[{{}, 1}] + 1
    local ex_height = ex_boxes[{{}, 4}] - ex_boxes[{{}, 2}] + 1
    local ex_ctr_x  = ex_boxes[{{},1}] + 0.5 * (ex_width - 1) 
    local ex_ctr_y  = ex_boxes[{{},2}] + 0.5 * (ex_height -1) 
    
    
    -- calculates width/height and center point of gt_boxes
    local gt_width = gt_boxes[{{}, 3}] - gt_boxes[{{}, 1}] + 1
    local gt_height = gt_boxes[{{}, 4}] - gt_boxes[{{}, 2}] + 1
    local gt_ctr_x  = gt_boxes[{{}, 1}] + 0.5 *(gt_width -1)
    local gt_ctr_y  = gt_boxes[{{}, 2}] + 0.5 *(gt_height -1) 

    --print(gt_ctr_x:add(-ex_ctr_x):cmul(torch.pow(ex_width,-1)))
    -- calculates regression
    gt_ctr_x:csub(ex_ctr_x):cdiv(ex_width + eps)
    gt_ctr_y:csub(ex_ctr_y):cdiv(ex_height + eps)
    gt_width:cdiv(ex_width):log()
    gt_height:cdiv(ex_height):log()    
    
    
    if torch.sum(gt_height:ne(gt_height)) > 0 then
          print('targets_dh')
          print(torch.cmul(gt_height,(torch.pow(ex_height, -1))))
          print(torch.log(torch.cmul(gt_height,(torch.pow(ex_height, -1)))))
    end
    
    regression_label = torch.cat(gt_ctr_x, gt_ctr_y, 2)
    regression_label = torch.cat(regression_label, gt_width, 2)
    regression_label = torch.cat(regression_label, gt_height, 2)
    
    return(regression_label)
end




--[[
function bbox_transform (ex_boxes, gt_boxes)
    -- calculates width/ height and center point of ex_boxes
    local ex_width = torch.add(torch.csub(ex_boxes[{{}, 3}], ex_boxes[{{}, 1}]), 1)
    local ex_height = torch.add(torch.csub(ex_boxes[{{}, 4}], ex_boxes[{{}, 2}]), 1)
    local ex_ctr_x  = ex_width:csub(1):mul(0.5):add(ex_boxes[{{},1}])
    local ex_ctr_y  = ex_height:csub(1):mul(0.5):add(ex_boxes[{{},2}])
    
    
    -- calculates width/height and center point of gt_boxes
    local gt_width = torch.add(torch.csub(gt_boxes[{{}, 3}], gt_boxes[{{}, 1}]), 1)
    local gt_height = torch.add(torch.csub(gt_boxes[{{}, 4}], gt_boxes[{{}, 2}]), 1)
    local gt_ctr_x  = gt_width:add(-1):mul(0.5):add(gt_width)
    local gt_ctr_y  = gt_height:add(-1):mul(0.5):add(gt_height)

    --print(gt_ctr_x:add(-ex_ctr_x):cmul(torch.pow(ex_width,-1)))
    -- calculates regression
    local targets_dx = torch.cmul(torch.add(gt_ctr_x, -ex_ctr_x), torch.pow(ex_width,-1))
    local targets_dy = torch.cmul(torch.add(gt_ctr_y, -ex_ctr_y), torch.pow(ex_height,-1))
    local targets_dw = torch.log(torch.cmul(gt_width,(torch.pow(ex_width, -1))))
    local targets_dh = torch.log(torch.cmul(gt_height,(torch.pow(ex_height, -1))))
    
    
    
    if torch.sum(targets_dh:ne(targets_dh)) > 0 then
          print('targets_dh')
          print(torch.cmul(gt_height,(torch.pow(ex_height, -1))))
          print(torch.log(torch.cmul(gt_height,(torch.pow(ex_height, -1)))))
    end
    
    regression_label = torch.cat(targets_dx, targets_dy, 2)
    regression_label = torch.cat(regression_label, targets_dw, 2)
    regression_label = torch.cat(regression_label, targets_dh, 2)
    
    return(regression_label)
end
]]--