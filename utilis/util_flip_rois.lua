function flip_rois(gt, im_size)
    local temp  = torch.Tensor(gt:size(1))
    temp[{{}}]:copy(gt:index(2,torch.LongTensor{4}))
    gt[{{},4}] =  torch.abs(torch.add(gt[{{},2}], -im_size[2])) + 1
    gt[{{},2}] =  torch.abs(torch.add(temp, -im_size[2])) + 1
    
    return gt
end