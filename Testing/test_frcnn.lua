--[[
test_results: table of i images
  -- bounding box 
  -- label of bounding boxes
  -- scores of bounding boxes
ground_truth:
  -- labels
  -- difficulty
  -- boxes
]]--
require('torch')
require('nn')

dofile('utilis/util_binaryToIdx.lua')
dofile('utilis/util_boxoverlap.lua')

torch.setdefaulttensortype('torch.FloatTensor')

gt = torch.load('/data/ethierer/ObjectDetection/FasterRCNN/Data/TestData/trainVal.t7')
ex = torch.load('/data/ethierer/ObjectDetection/FasterRCNN/Results/fastrcnn_1/Res50_Tue11Oct_8_33_ep16.t7/TueOct1121:34:352016/results.t7')

function test_frcn (test_results, ground_truth) 
  print(#ground_truth)
  print(#test_results)
  assert(#test_results == #ground_truth)
  
  local data = test_data 
  local num_classes = 20
  
  local false_positves = torch.Tensor(num_classes):fill(0)
  local true_positives = torch.Tensor(num_classes):fill(0)
  local npos = torch.Tensor(num_classes)
  
  for i = 1, #test_results do
    local found_idx = torch.zeros(ground_truth[i].labels:numel())
    
    local scores, sort_idx = torch.sort(test_results[i][3], 1, true)
    sort_idx:resize(sort_idx:numel())
    
    local ex_boxes = test_results[i][1]:index(1, sort_idx)
    local ex_labels = test_results[i][2]:index(1, sort_idx)
    
    local target_boxes = ground_truth[i].gt_boxes
    local labels = ground_truth[i].labels
    --local diff_boxes = ground_truth[i].diff
    
    local overlap = boxoverlap(ex_boxes, ground_truth[i].gt_boxes)
    
    local max, max_idx = torch.max(overlap, 2)
    max_idx:resize(max_idx:numel())
    max:resize(max:numel())
    
    local target_labels = labels:index(1, max_idx)
    target_labels:resize(target_labels:numel())
    --diff_boxes:index(1, max_idx)
    --diff_boxes:resize(diff_boxes:numel())
    
    local no_assignment = torch.zeros(ex_boxes:size(1))
    
    -- calculate all true positives
    for j = 1, ex_boxes:size(1) do
      if max[j] > 0.5 then
        --if diff_boxes[max_idx[j]] == 1 then
        --  no_assignment[j] = 0
        if target_labels[j] == ex_labels[j] then
          if found_idx[max_idx[j]] == 0 then
            found_idx[max_idx[j]] = 1
            true_positives[target_labels[j]] = true_positives[target_labels[j]] + 1
          else
            no_assignment[j] = 1
          end 
        else
          no_assignment[j] = 1
        end
      else
        no_assignment[j] = 1
      end
    end
    
    -- calculate false positives
    local nr_no_a = torch.sum(no_assignment)
    if nr_no_a > 0 then
      local idx_no_a = binaryToIdx(no_assignment:byte())
       
      for j = 1, nr_no_a do
        local cls = target_labels[idx_no_a[j]]
        false_positves[cls] = false_positves[cls] + 1
      end
    end
    
    -- calculate number of gt_boxes per label
    for j = 1, target_labels:size(1) do
      --if diff_boxes[j] == 0 then
        npos[target_labels[j]] = npos[target_labels[j]] + 1
      --end
    end
  end
  
  local recall = torch.cdiv(true_positives, npos) * 100
  local precision = torch.cdiv(true_positives, false_positves + true_positives) * 100
  return recall, precision
  
end


r,p = test_frcn(ex, gt)

print('Recall: ')
print(r)

print('Precision: ')
print(p)  