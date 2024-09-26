CREATE OR REPLACE TABLE `cameltrain.sight_logs.{log_id}_boundary_named_var_log` AS (
WITH
SimulationBoundaryState AS (
  SELECT
    location,
    array_reverse(ancestor_start_location)[offset(1)] AS parent_loc,
  FROM `cameltrain.sight_logs.{log_id}_log`
  WHERE sub_type='ST_BLOCK_START' AND block_start.sub_type = 'ST_SIMULATION_BOUNDARY_STATE'
),
SimulationTimeStep AS (
  SELECT
    location,
    block_start.label,
    array_reverse(ancestor_start_location)[offset(1)] AS parent_loc,
    block_start.simulation_time_step_start.time_step,
    block_start.simulation_time_step_start.time_step_index,
    block_start.simulation_time_step_start.time_step_size,
    block_start.simulation_time_step_start.time_step_units
  FROM `cameltrain.sight_logs.{log_id}_log`
  WHERE sub_type='ST_BLOCK_START' AND block_start.sub_type = 'ST_SIMULATION_TIME_STEP'
),
Simulation AS (
  SELECT
    location,
    array_reverse(ancestor_start_location)[offset(1)] AS parent_loc,
  FROM `cameltrain.sight_logs.{log_id}_log`
  WHERE sub_type='ST_BLOCK_START' AND block_start.sub_type = 'ST_SIMULATION'
),  
NamedVar AS (
  SELECT
    location,
    block_start.label,
    array_reverse(ancestor_start_location)[offset(1)] AS parent_loc
  FROM `cameltrain.sight_logs.{log_id}_log`
  WHERE sub_type='ST_BLOCK_START' AND block_start.sub_type = 'ST_NAMED_VALUE'
)
SELECT
  Simulation.location AS sim_location,
  SimulationBoundaryState.location AS sim_state_location,
  NamedVar.location AS named_var_location,
  NamedVar.label,
  time_step,
  CAST((SELECT STRING_AGG(CAST(ts AS STRING), ';') FROM UNNEST(time_step_index) ts) AS INT64) AS time_step_index,
  time_step_size,
  time_step_units,
FROM
  NamedVar
JOIN
  SimulationBoundaryState
ON SimulationBoundaryState.location = NamedVar.parent_loc
JOIN
  SimulationTimeStep
ON SimulationTimeStep.location = SimulationBoundaryState.parent_loc
JOIN
  Simulation
ON Simulation.location = SimulationTimeStep.parent_loc
)