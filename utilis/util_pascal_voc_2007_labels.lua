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

function loadLabelFromNumber(number)
  if number == 1 then
    return "aeroplane";
  elseif number == 2 then
    return "bicycle"
  elseif number == 3 then
    return "boat"
  elseif number == 4 then
    return "bus"
  elseif number == 5 then
    return "car"
  elseif number == 6 then
    return "cat"
  elseif number == 7 then
    return "chair"
  elseif number == 8 then
    return "diningtable"
  elseif number == 9 then
    return "dog"
  elseif number == 10 then
    return "horse"
  elseif number == 11 then
    return "motorbike"
  elseif number == 12 then
    return "pottedplant"
  elseif number == 13 then
    return "sofa"
  elseif number == 14 then
    return "train"
  elseif number == 15 then
    return "tvmonitor"
  elseif number == 16 then
    return "bird"
  elseif number == 17 then
    return "bottle"
  elseif number == 18 then
    return "cow"
  elseif number == 19 then
    return "person"
  elseif number == 20 then
    return "sheep"
  else 
    return "background"
  end
  
end