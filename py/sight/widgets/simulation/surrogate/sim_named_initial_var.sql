CREATE OR REPLACE TABLE `cameltrain.sight_logs.{log_label}_{log_id}_initial_named_var_log` AS (
WITH
SimulationInitialState AS (
  SELECT
    location,
    array_reverse(ancestor_start_location)[offset(1)] AS parent_loc,
  FROM `cameltrain.sight_logs.{log_label}_{log_id}_log`
  WHERE sub_type='ST_BLOCK_START' AND block_start.sub_type = 'ST_SIMULATION_INITIAL_STATE'
),
Simulation AS (
  SELECT
    location,
    array_reverse(ancestor_start_location)[offset(1)] AS parent_loc,
  FROM `cameltrain.sight_logs.{log_label}_{log_id}_log`
  WHERE sub_type='ST_BLOCK_START' AND block_start.sub_type = 'ST_SIMULATION'
),  
NamedVar AS (
  SELECT
    location,
    block_start.label,
    array_reverse(ancestor_start_location)[offset(1)] AS parent_loc
  FROM `cameltrain.sight_logs.{log_label}_{log_id}_log`
  WHERE sub_type='ST_BLOCK_START' AND block_start.sub_type = 'ST_NAMED_VALUE'
)
SELECT
  Simulation.location AS sim_location,
  SimulationInitialState.location AS sim_state_location,
  NamedVar.location AS named_var_location,
  NamedVar.label,
  0 AS time_step,
  0 AS time_step_index,
  0 AS time_step_size,
  NULL AS time_step_units,
FROM
  NamedVar
JOIN
  SimulationInitialState
ON SimulationInitialState.location = NamedVar.parent_loc
JOIN
  Simulation
ON Simulation.location = SimulationInitialState.parent_loc
)