--[[ Copyright (C) 2018 Google Inc.

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program; if not, write to the Free Software Foundation, Inc.,
51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
]]

-- Demonstration of creating a fixed level described using text.

-- local maze_generation = require 'dmlab.system.maze_generation'
-- local tensor = require 'dmlab.system.tensor'
-- local log = require 'common.log'
-- local random = require 'common.random'


local make_map = require 'common.make_map'
local pickups = require 'common.pickups'
local texture_sets = require 'themes.texture_sets'
local custom_observations = require 'decorators.custom_observations'
local setting_overrides = require 'decorators.setting_overrides'
local api = {}

--[[ Text map contents:

'P' - Player spawn point. Player is spawned with random orientation.
'A' - Apple pickup. 1 reward point when picked up.
'G' - Goal object. 10 reward points and the level restarts.
'I' - Door. Open and closes West-East corridors.
'H' - Door. Open and closes North-South corridors.
'*' - Walls.

Lights are placed randomly through out and decals are randomly placed on the
walls according to the theme.
]]
local TEXT_MAP = [[
*********************
*PPPPPPPPPPPPPPPPPPP*
*PPPPP*PPPPPPP*PPP*P*
*PPPPPPPPPPPPPPP*PPP*
*PPP*PPPPPPPPPP*PPPP*
*P*PPPPPPPPPPPPPPPPP*
*PPPPPPPPPPP**PPPPP**
*PPPPP*P*PPP*PPPPPPP*
*PPPPP*PPPPPPP*PPP*P*
*PPP*PPPPP*PPPPPPPPP*
*PP*PPPPPPPPP*PPPP*P*
*PPPPPPPPP*PPPPPPPPP*
*PPP*PPPPP*PP * P *P*
*P*PPPPPPPPP*      P*
*PP**PPP*P**   **   *
*PPPPPPPPP     G    *
**PPPPPPPPP *    *  *
*PPPPPPPPPPP       P*
*P*PPPPPPPP*P  * *PP*
*PPPPPPP*PPPPP P PPP*
*********************
]]

-- Called only once at start up. Settings not recognised by DM Lab internal
-- are forwarded through the params dictionary.
function api:init(params)
  -- Seed the map so only one map is created with lights and decals placed in
  -- the same place each run.
  make_map.random():seed(1)
  api._map = make_map.makeMap{
      mapName = "openfield_map_fixed",
      mapEntityLayer = TEXT_MAP,
      useSkybox = true,
      textureSet = texture_sets.TETRIS
  }
end

-- `make_map` has default pickup types A = apple_reward and G = goal.
-- This callback is used to create pickups with those names.
function api:createPickup(classname)
  return pickups.defaults[classname]
end

-- On first call we return the name of the map. On subsequent calls we return
-- an empty string. This informs the engine to only perform a quik map restart
-- instead.
function api:nextMap()
  local mapName = api._map
  api._map = ''
  return mapName
end

custom_observations.decorate(api)
setting_overrides.decorate{
    api = api,
    apiParams = {episodeLengthSeconds = 10 * 60, camera = {750, 750, 750}},
    decorateWithTimeout = true
}
return api
