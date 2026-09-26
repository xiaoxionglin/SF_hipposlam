-- Fixed native 11x11 random-maze map with optional visual landmarks.
local generator = require 'common.fixed_random_maze'
local landmarks = require 'common.easy_landmarks'
local random = require 'common.random'
local make_map = require 'common.make_map'
local tensor = require 'dmlab.system.tensor'
local custom_observations = require 'decorators.custom_observations'
local setting_overrides = require 'decorators.setting_overrides'

local api = {}

local function cueManifest(sites, mode, seed)
  local rows = {'schema\teasy-landmark-maze/v2', 'shape\t11\t11',
                'mode\t' .. mode, 'seed\t' .. seed,
                'cue_id\ttype\tasset\twall_row\twall_col\tfloor_row\tfloor_col\torientation'}
  for _, site in ipairs(sites) do
    rows[#rows + 1] = table.concat({site.cueId, site.cueType, site.asset,
      site.wallRow - 1, site.wallCol - 1, site.floorRow - 1, site.floorCol - 1,
      site.orientation}, '\t')
  end
  return table.concat(rows, '\n') .. '\n'
end

local function solid(red, green, blue)
  local result = tensor.ByteTensor(64, 64, 4)
  result:select(3, 1):fill(red)
  result:select(3, 2):fill(green)
  result:select(3, 3):fill(blue)
  result:select(3, 4):fill(255)
  return result
end

function api:init(params)
  local geometrySeed = assert(tonumber(params.geometrySeed), 'geometrySeed required')
  local opening = assert(tonumber(params.wallRemovalProbability), 'wallRemovalProbability required')
  local mapRows = assert(tonumber(params.mapRows), 'mapRows required')
  local mapCols = assert(tonumber(params.mapCols), 'mapCols required')
  local cueSeed = assert(tonumber(params.cueLayoutSeed), 'cueLayoutSeed required')
  local cueMode = assert(params.landmarkCues, 'landmarkCues required')
  assert(geometrySeed == 1001, 'easy landmark maze is fixed to geometry seed 1001')
  assert(opening == 0.85, 'easy landmark maze requires 85 percent wall removal')
  assert(mapRows == 11 and mapCols == 11, 'easy landmark maze requires native 11x11 dimensions')
  assert(cueSeed == 20260923, 'unreviewed cue layout seed')
  assert(cueMode == 'rich' or cueMode == 'none', 'landmarkCues must be rich or none')
  random:seed(geometrySeed)
  self._entity, self._spawnAnchor = generator.generate(mapRows, mapCols, random, opening)
  self._sites = landmarks.sites(self._entity, cueSeed)
  self._cueMode = cueMode
  self._cueManifest = cueManifest(self._sites, cueMode, cueSeed)
  make_map.random():seed(1)
  self._map = make_map.makeMap{
    mapName = 'elm1_' .. geometrySeed .. '_11x11_q85_' .. (cueMode == 'rich' and 'r' or 'n'),
    mapEntityLayer = self._entity,
    mapVariationsLayer = landmarks.variations(self._entity, self._sites),
    useSkybox = true,
    theme = landmarks.theme(self._sites, cueMode == 'rich'),
  }
  custom_observations.addSpec(
      'GEOMETRY.ENTITY_LAYER', 'String', {0}, function() return self._entity end)
  custom_observations.addSpec(
      'GEOMETRY.CUE_MANIFEST', 'String', {0}, function() return self._cueManifest end)
end

function api:loadTexture(textureName)
  if textureName:find('easy_landmark/neutral', 1, true) then return solid(112, 112, 112) end
  local index = tonumber(textureName:match('easy_landmark/color(%d%d)'))
  if index then
    local color = landmarks.colors()[index]
    assert(color, 'unknown landmark color texture')
    return solid(color[1], color[2], color[3])
  end
end

function api:start(episode, seed) random:seed(seed) end

function api:nextMap()
  api.setInstruction('3')
  return self._map
end

function api:hasEpisodeFinished(timeSeconds) return false end

custom_observations.decorate(api)
setting_overrides.decorate{
  api = api,
  apiParams = {
    geometrySeed = 1001,
    wallRemovalProbability = 0.85,
    mapRows = 11,
    mapCols = 11,
    geometryHash = '',
    cueLayoutSeed = 20260923,
    cueLayoutHash = '',
    landmarkCues = 'rich',
    episodeLengthSeconds = 120,
    camera = {1050, 1050, 1000},
  },
  decorateWithTimeout = true,
}
return api
