-- Review-only cue approach renderer. Production uses easy_landmark_maze_noreward.
local generator = require 'common.fixed_random_maze'
local landmarks = require 'common.easy_landmarks'
local random = require 'common.random'
local make_map = require 'common.make_map'
local tensor = require 'dmlab.system.tensor'
local setting_overrides = require 'decorators.setting_overrides'

local api = {}

local function solid(red, green, blue)
  local result = tensor.ByteTensor(64, 64, 4)
  result:select(3, 1):fill(red)
  result:select(3, 2):fill(green)
  result:select(3, 3):fill(blue)
  result:select(3, 4):fill(255)
  return result
end

local function previewEntity(entity, site)
  local rows = {}
  local rowIndex = 0
  for row in entity:gmatch('[^\n]+') do
    rowIndex = rowIndex + 1
    local chars = {}
    for column = 1, #row do
      local value = row:sub(column, column)
      chars[column] = value == 'P' and ' ' or value
    end
    if rowIndex == site.floorRow then chars[site.floorCol] = 'P' end
    rows[rowIndex] = table.concat(chars)
  end
  return table.concat(rows, '\n') .. '\n'
end

function api:init(params)
  local cue = assert(tonumber(params.previewCue), 'previewCue required')
  assert(cue >= 1 and cue <= 20, 'previewCue must be in [1, 20]')
  random:seed(1001)
  local entity = generator.generate(11, 11, random, 0.85)
  local sites = landmarks.sites(entity, 20260923)
  self._site = sites[cue]
  self._map = make_map.makeMap{
    mapName = 'elm_11x11_q85_preview_' .. cue,
    mapEntityLayer = previewEntity(entity, self._site),
    mapVariationsLayer = landmarks.variations(entity, sites),
    useSkybox = true,
    theme = landmarks.theme(sites, true),
  }
end

function api:loadTexture(textureName)
  if textureName:find('easy_landmark/neutral', 1, true) then return solid(112, 112, 112) end
  local index = tonumber(textureName:match('easy_landmark/color(%d%d)'))
  if index then
    local color = landmarks.colors()[index]
    return solid(color[1], color[2], color[3])
  end
end

function api:nextMap() return self._map end

function api:updateSpawnVars(spawnVars)
  if spawnVars.classname == 'info_player_start' then
    local angles = {north = '90', east = '0', south = '270', west = '180'}
    spawnVars.angle = angles[self._site.orientation]
    spawnVars.randomAngleRange = '0'
  end
  return spawnVars
end

setting_overrides.decorate{
  api = api,
  apiParams = {previewCue = 1, episodeLengthSeconds = 10, camera = {1050, 1050, 1000}},
  decorateWithTimeout = true,
}
return api
