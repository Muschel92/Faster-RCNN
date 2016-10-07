
function img_from_mean(img, mean)
    
    img[1]:add(mean[1])
    img[2]:add(mean[2])
    img[3]:add(mean[3])
    
    return img
end
