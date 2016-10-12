-- calculates the mean Image of all train Images
---------------------------------------------------

require 'torch'
require 'image'

---------------------------------------------------

imdb_train = torch.load('train_roidb.t7', 'ascii')
print("Loaded Image Infos")

meanImage = torch.zeros(3, 1000, 1000)

local counter = 0

for i in ipairs(imdb_train) do
  im = image.load(imdb_train[i].path, 3, 'byte')
  
  im = image.scale(im, 1000, 1000)
  
  meanImage = torch.add(meanImage, im:double())
  
  counter = counter + 1
end

meanImage = torch.div(meanImage, counter)
meanImage = meanImage:byte()exit

torch.save('meanImage.t7', meanImage)


