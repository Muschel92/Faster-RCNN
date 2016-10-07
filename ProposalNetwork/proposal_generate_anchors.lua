-- calculates all the base anchors for the different ratios and scales
-- returns all anchors[ratio][scale][xmin, ymin, xmax, ymax] as tensor
function proposal_generate_anchors (conf)

    local base_anchor = torch.Tensor{ 1, 1, conf.anchor_base_size+1, conf.anchor_base_size+1}
    local ratio_anchors = ratio_jitter(base_anchor, conf.anchor_ratios)

    anchors = torch.Tensor(ratio_anchors:size(1), conf.anchor_scales:size(1), 4)
    for i = 1,ratio_anchors:size()[1] do
      local scale = conf.anchor_scales
      anchors[{i, {},{}}] = scale_jitter(ratio_anchors[{i, {}}], scale)
    end
    anchors = anchors:reshape(anchors:size(1)* anchors:size(2), 4)
    return anchors
end


-- calculates rectangle of anchor for all ratios
-- returns xmin, ymin, xmax, ymax for all ratios
function ratio_jitter(anchor, ratios)
    local ratio = ratios
    --print(ratio)
    
    local w = anchor[3] - anchor[1]
    --print("W: " .. w)
    local h = anchor[4] - anchor[2]
    --print("H: " .. h)
    
    -- calculate the center coordinates of anchor
    local x_ctr = anchor[1] + w / 2
    local y_ctr = anchor[2] + h / 2

    local size = w * h
    
    -- calculate width and hight of different ratios
    local anchors =  torch.Tensor( ratio:size(1), 4)

    local wr = torch.round(torch.pow(ratio, 0.5))
    local hr = torch.round(torch.cdiv(ratio, wr ))
    
    -- x_ctr - (ws - 1) / 2
    anchors[{{}, 1}] = -torch.div(wr, 2)
    --anchors[{{}, 1}] = torch.add(-torch.div(wr, 2), x_ctr)
    -- y_ctr - (hs - 1) /2
    anchors[{{}, 2}] = -torch.div(hr, 2)
    --anchors[{{}, 2}] = torch.add(-torch.div(hr, 2), y_ctr)
    -- x_ctr + (ws - 1)/ 2
    anchors[{{}, 3}] = torch.div(wr, 2)
    --anchors[{{}, 3}] = torch.add(torch.div(wr, 2), x_ctr)
    -- y_ctr + (hs - 1)/ 2
    anchors[{{}, 4}] = torch.div(hr, 2)
    --anchors[{{}, 4}] = torch.add(torch.div(hr, 2), y_ctr)

    return anchors    
    
end

-- calculates rectangle of anchor for all scales
-- returns xmin, ymin, xmax, ymax for all scales
function scale_jitter (anchor, scales)
  
    
    local scale = scales
    
    local w = anchor[3] - anchor[1]
    local h = anchor[4] - anchor[2]
    local size = w*h
    
    local x_ctr = anchor[1] + w / 2
    local y_ctr = anchor[2] + w / 2

    local anchors = torch.Tensor(scale:size(1), 4)
    
    local ws = torch.mul(scale[{{},1}], w)
    --torch.round(torch.mul(torch.pow(torch.mul(scale, w), -1), size))
    local hs = torch.mul(scale[{{},2}], h)
    --torch.round(torch.mul(torch.pow(ws, -1), size))
    
    local zero = torch.zeros(ws:size(1))
    -- x_ctr - (ws - 1) / 2
    anchors[{{}, 1}] = torch.add(-torch.div(ws, 2), x_ctr)
    -- y_ctr - (ws - 1) /2
    anchors[{{}, 2}] = torch.add(-torch.div(hs, 2), y_ctr)
    -- x_ctr + (ws - 1)/ 2
    anchors[{{}, 3}] = torch.add(torch.div(ws, 2), x_ctr)
    -- y_ctr + (ws - 1)/ 2
    anchors[{{}, 4}] = torch.add(torch.div(hs, 2), y_ctr)
    
    return anchors    
end


