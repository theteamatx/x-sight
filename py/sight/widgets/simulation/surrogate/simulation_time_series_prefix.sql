WITH
Values AS (
SELECT
  location,
  sub_type,
  value.double_value,
  ancestor_start_location,
  array_reverse(ancestor_start_location)[offset(1)] AS parent_loc
FROM `cameltrain.sight_logs.{log_label}_{log_id}_log`
WHERE sub_type='ST_VALUE'
),
NamedVar AS (
  SELECT
    location,
    block_start.label,
    array_reverse(ancestor_start_location)[offset(1)] AS parent_loc
  FROM `cameltrain.sight_logs.{log_label}_{log_id}_log`
  WHERE sub_type='ST_BLOCK_START' AND block_start.sub_type = 'ST_NAMED_VALUE'
),
SimulationState AS (
  SELECT
    location,
    array_reverse(ancestor_start_location)[offset(1)] AS parent_loc,
  FROM `cameltrain.sight_logs.{log_label}_{log_id}_log`
  WHERE sub_type='ST_BLOCK_START' AND block_start.sub_type = 'ST_SIMULATION_STATE'
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
  FROM `cameltrain.sight_logs.{log_label}_{log_id}_log`
  WHERE sub_type='ST_BLOCK_START' AND block_start.sub_type = 'ST_SIMULATION_TIME_STEP'
),
Simulation AS (
  SELECT
    location,
    array_reverse(ancestor_start_location)[offset(1)] AS parent_loc,
  FROM `cameltrain.sight_logs.{log_label}_{log_id}_log`
  WHERE sub_type='ST_BLOCK_START' AND block_start.sub_type = 'ST_SIMULATION'
),
Data AS (
SELECT
  Simulation.location AS sim_location,
  NamedVar.location,
  NamedVar.label,
  double_value,
  time_step,
  -- ARRAY_TO_STRING(time_step_index) AS time_step_index,
  CAST((SELECT STRING_AGG(CAST(ts AS STRING), ';') FROM UNNEST(time_step_index) ts) AS INT64) AS time_step_index,
  time_step_size,
  time_step_units,
FROM
  Values
JOIN
  NamedVar
ON NamedVar.location = Values.parent_loc
JOIN
  SimulationState
ON SimulationState.location = NamedVar.parent_loc
JOIN
  SimulationTimeStep
ON SimulationTimeStep.location = SimulationState.parent_loc
JOIN
  Simulation
ON Simulation.location = SimulationTimeStep.parent_loc
ORDER BY NamedVar.location, label
),
AllVars AS (
  SELECT label FROM Data GROUP BY label ORDER BY label
)