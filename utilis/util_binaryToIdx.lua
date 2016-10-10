function binaryToIdx(t)
  local r = torch.range(1, t:size(1)):long()
  r = r[t]
  
  return r
end
