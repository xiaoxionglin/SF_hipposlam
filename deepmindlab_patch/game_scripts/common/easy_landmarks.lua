-- Deterministic visual landmarks for a fixed native entity map.
local M = {}

local DIRECTIONS = {
  {name = 'north', code = 'N', wallDr = -1, wallDc = 0},
  {name = 'east', code = 'E', wallDr = 0, wallDc = 1},
  {name = 'south', code = 'S', wallDr = 1, wallDc = 0},
  {name = 'west', code = 'W', wallDr = 0, wallDc = -1},
}
local DECALS = {}
for index = 1, 10 do
  DECALS[index] = string.format('decal/lab_games/dec_img_style01_%03d', index)
end
local COLORS = {
  {230, 25, 75}, {60, 180, 75}, {67, 99, 216}, {245, 130, 49}, {145, 30, 180},
  {66, 212, 244}, {240, 50, 230}, {191, 239, 69}, {250, 190, 212}, {70, 153, 144},
}

local function lines(entity)
  local result = {}
  for line in entity:gmatch('[^\n]+') do result[#result + 1] = line end
  return result
end

local function charAt(rows, row, col)
  return rows[row]:sub(col, col)
end

local function shuffle(values, seed)
  local state = seed % 4294967296
  for index = #values, 2, -1 do
    state = (1664525 * state + 1013904223) % 4294967296
    local selected = state % index + 1
    values[index], values[selected] = values[selected], values[index]
  end
end

function M.sites(entity, seed)
  local rows = lines(entity)
  local candidates = {}
  for wallRow = 1, #rows do
    for wallCol = 1, #rows[1] do
      if charAt(rows, wallRow, wallCol) == '*' then
        for _, direction in ipairs(DIRECTIONS) do
          local floorRow = wallRow - direction.wallDr
          local floorCol = wallCol - direction.wallDc
          if floorRow >= 1 and floorRow <= #rows and
              floorCol >= 1 and floorCol <= #rows[1] and
              charAt(rows, floorRow, floorCol) ~= '*' then
            candidates[#candidates + 1] = {
              wallRow = wallRow, wallCol = wallCol,
              floorRow = floorRow, floorCol = floorCol,
              orientation = direction.name, direction = direction.code,
            }
          end
        end
      end
    end
  end
  shuffle(candidates, seed)
  local chosen, walls, floors = {}, {}, {}
  for _, candidate in ipairs(candidates) do
    local wallKey = candidate.wallRow .. ':' .. candidate.wallCol
    local floorKey = candidate.floorRow .. ':' .. candidate.floorCol
    if not walls[wallKey] and not floors[floorKey] then
      walls[wallKey], floors[floorKey] = true, true
      local index = #chosen + 1
      candidate.cueType = index <= 10 and 'decal' or 'color'
      candidate.typeIndex = index <= 10 and index or index - 10
      candidate.cueId = string.format(index <= 10 and 'D%02d' or 'C%02d', candidate.typeIndex)
      candidate.asset = candidate.cueType == 'decal' and DECALS[candidate.typeIndex]
          or string.format('#%02X%02X%02X', unpack(COLORS[candidate.typeIndex]))
      chosen[#chosen + 1] = candidate
      if #chosen == 20 then break end
    end
  end
  assert(#chosen == 20, 'map does not contain 20 distinct visible cue sites')
  return chosen
end

local function setChar(row, column, value)
  return row:sub(1, column - 1) .. value .. row:sub(column + 1)
end

function M.variations(entity, sites)
  local rows = lines(entity)
  for row = 1, #rows do rows[row] = rows[row]:gsub('.', '.') end
  local colored = 0
  for _, site in ipairs(sites) do
    if site.cueType == 'color' then
      rows[site.floorRow] = setChar(rows[site.floorRow], site.floorCol,
          string.char(string.byte('A') + site.typeIndex - 1))
      colored = colored + 1
    end
  end
  assert(colored == 10, 'expected exactly ten colored landmark wall faces')
  return table.concat(rows, '\n') .. '\n'
end

function M.theme(sites, rich)
  local neutral = {tex = 'easy_landmark/neutral', width = 64, height = 64}
  local floor = {tex = 'map/lab_games/lg_style_02_floor_blue'}
  local ceiling = {tex = 'map/lab_games/lg_style_02_ceiling_blue'}
  local byVariation = {}
  for _, site in ipairs(sites) do
    if site.cueType == 'color' then byVariation[string.char(string.byte('A') + site.typeIndex - 1)] = site end
  end
  local theme = {}
  function theme:mazeVariation(variation)
    local result = {floor = floor, ceiling = ceiling, wallN = neutral, wallE = neutral,
                    wallS = neutral, wallW = neutral}
    local site = rich and byVariation[variation] or nil
    if site then result['wall' .. site.direction] = {
      tex = 'easy_landmark/color' .. string.format('%02d', site.typeIndex), width = 64, height = 64,
    } end
    return result
  end
  function theme:placeWallDecals(locations)
    if not rich then return {} end
    local wanted = {}
    for _, site in ipairs(sites) do
      if site.cueType == 'decal' then
        wanted[site.floorRow .. ':' .. site.floorCol .. ':' .. site.direction] = site
      end
    end
    local result = {}
    for _, location in ipairs(locations) do
      local site = wanted[location.i .. ':' .. location.j .. ':' .. location.direction]
      if site then result[#result + 1] = {
        index = location.index, decal = {tex = site.asset .. '_nonsolid'},
      } end
    end
    assert(#result == 10, 'expected exactly ten landmark decals, got ' .. #result)
    return result
  end
  return theme
end

function M.colors() return COLORS end
return M
