-- Fixed-map extraction of levels/demos/random_maze.lua.
-- Preserves the original RNG call order for carving, wall removal, anchor
-- selection, and flood-distance spawn placement.
local maze_generation = require 'dmlab.system.maze_generation'
local tensor = require 'dmlab.system.tensor'

local M = {}

local function getRandomEvenCoordinate(rows, cols, random)
  assert(rows > 2 and rows % 2 == 1)
  assert(cols > 2 and cols % 2 == 1)
  return {
      random:uniformInt(1, math.floor(rows / 2)) * 2,
      random:uniformInt(1, math.floor(cols / 2)) * 2,
  }
end

local function findVisitableCells(row, column, mazeTensor)
  local shape = mazeTensor:shape()
  local visitable = {}
  if row - 2 > 1 and mazeTensor(row - 2, column):val() == 0 then
    visitable[#visitable + 1] = {{row - 2, column}, {row - 1, column}}
  end
  if column - 2 > 1 and mazeTensor(row, column - 2):val() == 0 then
    visitable[#visitable + 1] = {{row, column - 2}, {row, column - 1}}
  end
  if row + 2 < shape[1] and mazeTensor(row + 2, column):val() == 0 then
    visitable[#visitable + 1] = {{row + 2, column}, {row + 1, column}}
  end
  if column + 2 < shape[2] and mazeTensor(row, column + 2):val() == 0 then
    visitable[#visitable + 1] = {{row, column + 2}, {row, column + 1}}
  end
  return visitable
end

local function generateTensorMaze(rows, cols, random)
  local mazeTensor = tensor.ByteTensor(rows, cols)
  local stack = {getRandomEvenCoordinate(rows, cols, random)}
  while #stack ~= 0 do
    local row, column = unpack(stack[#stack])
    mazeTensor(row, column):val(1)
    local visitable = findVisitableCells(row, column, mazeTensor)
    if #visitable > 0 then
      local choice = visitable[random:uniformInt(1, #visitable)]
      mazeTensor(unpack(choice[2])):val(1)
      stack[#stack + 1] = choice[1]
    else
      stack[#stack] = nil
    end
  end
  return mazeTensor
end

function M.generate(rows, cols, random, wallRemovalProbability)
  assert(wallRemovalProbability >= 0 and wallRemovalProbability <= 1)
  local mazeTensor = generateTensorMaze(rows, cols, random)
  local maze = maze_generation.mazeGeneration{height = rows, width = cols}
  mazeTensor:applyIndexed(function(value, index)
    local row, column = unpack(index)
    if 1 < row and row < rows and 1 < column and column < cols and
        random:uniformReal(0, 1) < wallRemovalProbability then
      maze:setEntityCell(row, column, ' ')
    else
      maze:setEntityCell(row, column, value == 0 and '*' or ' ')
    end
    maze:setVariationsCell(row, column, '.')
  end)

  -- Preserve random_maze.lua's post-geometry RNG call and flood-distance
  -- spawn rule. The anchor is temporary: this task has no goal or pickups.
  local anchor = getRandomEvenCoordinate(rows, cols, random)
  maze:setEntityCell(anchor[1], anchor[2], 'G')
  maze:visitFill{
      cell = anchor,
      func = function(row, column, distance)
        if distance > 5 then maze:setEntityCell(row, column, 'P') end
      end,
  }
  maze:setEntityCell(anchor[1], anchor[2], ' ')
  return maze:entityLayer(), anchor
end

return M
