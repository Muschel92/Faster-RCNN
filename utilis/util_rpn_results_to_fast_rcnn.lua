
-- calculates from the test output of rpn the new roidbs data base for
-- fast rcnn
function transform_rpn_data_to_fast_rcnn(roidbs, result)
    if (#roidbs ~= #result) then
      return
    end   
        
    for i = 1, #roidbs do
        
        print(i)
        -- calculate scores for every bbox:
        -- score = fg_score - bg_score
        local scores = result[i][1]
        --local scores = torch.add(result[i][1][{{},1}] , - result[i][1][{{},2}])
        
        -- calculate target boxes with nms

        
        local boxes_idx = nms(result[i][2]:float(), 0.7, scores:float())
        local boxes = result[i][2]:index(1, boxes_idx)
        
          
       -- add the groundtruth boxes
       --[[
        if (roidbs[i].boxes ~= nil) then
          boxes = roidbs[i].boxes:cuda():cat(boxes, 1)
        else
          boxes = roidbs[i].gt_boxes:cuda():cat(boxes, 1)
        end        
        ]]--
        
        roidbs[i].bbox_targets = nil
        roidbs[i].ex_boxes = boxes:float()
        if(roidbs[i].boxes  ~= nil) then
         roidbs[i].gt_boxes = roidbs[i].boxes:float()
         roidbs[i].boxes = nil
        end
        roidbs[i].scores = scores:float()
        roidbs[i].size = roidbs[i].size:float()
        roidbs[i].labels = roidbs[i].labels:float()
    end
    return roidbs
  
end
