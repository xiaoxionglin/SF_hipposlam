-- Connected depth-first maze carving extracted from levels/demos/random_maze.lua.
-- Original Copyright (C) 2018 Google Inc.; GPL-2.0-or-later.
-- The caller owns the RNG: geometry never consumes an episode's spawn stream.
local M = {}
function M.generate(rows, cols, random, opening)
  assert(rows > 2 and rows % 2 == 1 and cols > 2 and cols % 2 == 1)
  assert(opening >= 0 and opening <= 1)
  local floor = {}
  for r = 1, rows do
    floor[r] = {}
    for c = 1, cols do floor[r][c] = false end
  end
  local stack = {{random:uniformInt(1, math.floor(rows / 2)) * 2,
                  random:uniformInt(1, math.floor(cols / 2)) * 2}}
  local directions = {{-2, 0}, {0, -2}, {2, 0}, {0, 2}}
  while #stack > 0 do
    local r, c = stack[#stack][1], stack[#stack][2]
    floor[r][c] = true
    local choices = {}
    for _, d in ipairs(directions) do
      local rr, cc = r + d[1], c + d[2]
      if rr > 1 and rr < rows and cc > 1 and cc < cols and not floor[rr][cc] then
        choices[#choices + 1] = {rr, cc}
      end
    end
    if #choices == 0 then
      stack[#stack] = nil
    else
      local nextCell = choices[random:uniformInt(1, #choices)]
      floor[(r + nextCell[1]) / 2][(c + nextCell[2]) / 2] = true
      stack[#stack + 1] = nextCell
    end
  end
  -- Draw for every interior wall in fixed row-major order, independently of q.
  -- Retain the backbone component below, excluding isolated opened wall pockets.
  for r = 2, rows - 1 do
    for c = 2, cols - 1 do
      if not floor[r][c] then floor[r][c] = random:uniformReal(0, 1) < opening end
    end
  end
  -- A newly opened odd/odd wall can be an isolated pocket. Keep only the
  -- backbone-connected component. Reachability is monotone in opening, so
  -- this correction preserves nesting and never removes carved passages.
  local reached, queue = {}, {{2, 2}}
  reached[2 * cols + 2] = true
  local head = 1
  while head <= #queue do
    local cell = queue[head]; head = head + 1
    for _, d in ipairs({{-1,0},{1,0},{0,-1},{0,1}}) do
      local rr, cc = cell[1] + d[1], cell[2] + d[2]
      local key = rr * cols + cc
      if rr > 1 and rr < rows and cc > 1 and cc < cols and floor[rr][cc] and not reached[key] then
        reached[key] = true; queue[#queue + 1] = {rr, cc}
      end
    end
  end
  for rr = 2, rows - 1 do
    for cc = 2, cols - 1 do floor[rr][cc] = reached[rr * cols + cc] or false end
  end
  local lines = {}
  for r = 1, rows do
    local line = {}
    for c = 1, cols do
      line[c] = floor[r][c] and ' ' or '*'
      if r % 2 == 0 and c % 2 == 0 then line[c] = 'P' end
    end
    lines[r] = table.concat(line)
  end
  return table.concat(lines, '\n') .. '\n'
end
return M
