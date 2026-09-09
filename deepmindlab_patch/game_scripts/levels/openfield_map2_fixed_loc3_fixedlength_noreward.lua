local api = require 'levels.openfield_map2_fixed_loc3_noreward'

-- Goal contact is an observation event in this exploration level, not a terminal.
-- The base level's existing timeout decorator still ends the episode at 120 s.
function api:hasEpisodeFinished(timeSeconds)
  return false
end

return api
