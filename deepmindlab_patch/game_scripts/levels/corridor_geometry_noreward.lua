-- Fixed connected geometry; episode seeds affect spawning, never the map.
local generator = require 'common.connected_maze'
local random = require 'common.random'
local make_map = require 'common.make_map'
local texture_sets = require 'themes.texture_sets'
local custom_observations = require 'decorators.custom_observations'
local setting_overrides = require 'decorators.setting_overrides'
local api = {}
function api:init(params)
  local seed = assert(tonumber(params.geometrySeed), 'geometrySeed required')
  local opening = assert(tonumber(params.wallRemovalProbability), 'wallRemovalProbability required')
  assert(seed == math.floor(seed) and seed >= 0)
  random:seed(seed)
  self._entity = generator.generate(21, 21, random, opening)
  make_map.random():seed(1)
  self._map = make_map.makeMap{
    mapName = 'cg1_' .. seed .. '_' .. string.format('%.2f', opening):gsub('%.', '_')
      .. '_' .. (params.geometryHash or 'export'):sub(1, 16),
    mapEntityLayer = self._entity, useSkybox = true, textureSet = texture_sets.TETRIS,
  }
  custom_observations.addSpec('GEOMETRY.ENTITY_LAYER', 'String', {0}, function() return self._entity end)
end
function api:start(episode, seed) random:seed(seed) end
function api:nextMap()
  api.setInstruction('3') -- Preserve the existing constant context; no map identity input.
  return self._map
end
function api:hasEpisodeFinished(timeSeconds) return false end
custom_observations.decorate(api)
setting_overrides.decorate{
  api = api, apiParams = {geometrySeed = 1001, wallRemovalProbability = 0, geometryHash = 'export', episodeLengthSeconds = 120, camera = {1050, 1050, 1000}},
  decorateWithTimeout = true,
}
return api
