
function loadBboxes(xml_path, jpg_path)
 
  xml = require("LuaXml")
  
  local file = xml.load(xml_path)


  nr = #file - 6
  boxes = torch.Tensor(nr,4)
  diff = torch.Tensor(nr)
  labels = torch.Tensor(nr,1)
  
  local image = {}
  for i = 1,nr do
    
    if ( file[6+i][5]:find("name") ~= nil) then
      length =  #file[6+i]
      diff[i] = tonumber(file[6+i][4][1])
      boxes[i][2] = tonumber(file[6+i][length][1][1])
      boxes[i][1] = tonumber(file[6+i][length][2][1])
      boxes[i][4] = tonumber(file[6+i][length][3][1])
      boxes[i][3] = tonumber(file[6+i][length][4][1])
      labels[i][1] = loadLabelsPascal(file[6+i][1][1])
    else
      diff[i] = tonumber(file[6+i][4][1])
      boxes[i][2] = tonumber(file[6+i][5][1][1])
      boxes[i][1] = tonumber(file[6+i][5][2][1])
      boxes[i][4] = tonumber(file[6+i][5][3][1])
      boxes[i][3] = tonumber(file[6+i][5][4][1])
      labels[i][1] = loadLabelsPascal(file[6+i][1][1])
    end
  end
    
  image.gt_boxes = boxes
  image.labels = labels
  image.path = jpg_path .. file[2][1]
  local temp = string.split(file[2][1], "%.")
  image.size = torch.Tensor{tonumber(file[5][2][1]), tonumber(file[5][1][1])}
  image.id = temp[1]
  image.imdb_name = file[3][2][1]
  image.num_classes = 20
  image.diff = diff
  
  return(image)
  
end



function loadLabelsPascal(name)
  if (name == "aeroplane") then
    return (1);
  elseif (name == "bicycle") then
    return(2)
  elseif (name == "boat") then
    return(3)
  elseif(name == "bus") then
    return(4)
  elseif (name == "car") then
    return(5)
  elseif (name == "cat") then
    return(6)
  elseif (name == "chair") then
    return(7)
  elseif(name == "diningtable") then
    return(8)
  elseif (name == "dog") then
    return (9)
  elseif (name == "horse") then
    return(10)
  elseif (name == "motorbike") then
    return(11)
  elseif ( name == "pottedplant") then
    return(12)
  elseif (name == "sofa") then
    return (13)
  elseif (name == "train") then
    return (14)
  elseif (name == "tvmonitor") then
    return (15)
  elseif(name == "bird") then
    return (16)
  elseif(name == "bottle") then
    return (17)
  elseif(name == "cow") then
    return (18)
  elseif(name == "person") then
    return (19)
  elseif (name == "sheep") then
    return (20)
  else 
    return(0)
  end
  
end
