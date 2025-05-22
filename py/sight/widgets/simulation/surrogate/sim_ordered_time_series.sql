CREATE OR REPLACE TABLE `cameltrain.sight_logs.{log_label}_{log_id}_simulation_ordered_time_series_log` AS (
-- SELECT
--   sim_location,
--   time_step_index,
--   time_step_index+1 AS next_time_step_index,
--   double_values,
--   string_values,
--   value_types
-- FROM
--   `cameltrain.sight_logs.{log_label}_{log_id}_simulation_unordered_time_series_log`
-- ORDER BY sim_location, time_step_index

WITH 
AllCompleteAutoregObservations AS (
  SELECT
    * 
  FROM 
    `cameltrain.sight_logs.{log_label}_{log_id}_autoreg_simulation_unordered_time_series_log`
  WHERE 
    ARRAY_LENGTH(values)={num_autoreg_vars} 
),
AllCompleteBoundaryObservations AS (
  SELECT
    * 
  FROM 
    `cameltrain.sight_logs.{log_label}_{log_id}_boundary_simulation_unordered_time_series_log`
  WHERE 
    ARRAY_LENGTH(values)={num_boundary_vars} 
),
AllCompleteInitialObservations AS (
  SELECT
    * 
  FROM 
    `cameltrain.sight_logs.{log_label}_{log_id}_initial_simulation_unordered_time_series_log`
  WHERE 
    ARRAY_LENGTH(values)={num_initial_vars} 
),

AllCompleteAutoregRuns AS (
  SELECT
    sim_location,
    COUNT(DISTINCT time_step_index) AS num_time_steps
  FROM
    AllCompleteAutoregObservations
  GROUP BY
    sim_location
  ORDER BY 
    sim_location ASC
),
AllCompleteBoundaryRuns AS (
  SELECT
    sim_location,
    COUNT(DISTINCT time_step_index) AS num_time_steps
  FROM
    AllCompleteBoundaryObservations
  GROUP BY
    sim_location
  ORDER BY 
    sim_location ASC
),

NumTimeStepsPerRun AS (
  SELECT
    MAX(num_time_steps) AS max_time_steps
  FROM
  (
    SELECT * FROM AllCompleteAutoregRuns
    UNION ALL 
    SELECT * FROM AllCompleteBoundaryRuns
  )
),

AutoregRunsWithAllTimeSteps AS (
  SELECT
    sim_location
  FROM
    AllCompleteAutoregRuns
  JOIN
    NumTimeStepsPerRun
  ON
    AllCompleteAutoregRuns.num_time_steps = NumTimeStepsPerRun.max_time_steps
),
BoundaryRunsWithAllTimeSteps AS (
  SELECT
    sim_location
  FROM
    AllCompleteBoundaryRuns
  JOIN
    NumTimeStepsPerRun
  ON
    AllCompleteBoundaryRuns.num_time_steps = NumTimeStepsPerRun.max_time_steps
)


SELECT
  sim_location,
  AllCompleteAutoRegObservations.time_step_index,
  -- time_step_index+1 AS next_time_step_index,
  -- ARRAY_TO_STRING(labels, ", ") AS labels,
  ARRAY_TO_STRING(AllCompleteAutoRegObservations.values, ", ") AS autoreg_values,
  ARRAY_TO_STRING(AllCompleteBoundaryObservations.values, ", ") AS boundary_values,
  ARRAY_TO_STRING(AllCompleteInitialObservations.values, ", ") AS initial_values,
FROM
  AutoregRunsWithAllTimeSteps
JOIN
  BoundaryRunsWithAllTimeSteps
USING(sim_location)
JOIN
  AllCompleteAutoRegObservations
USING(sim_location)
JOIN
  AllCompleteBoundaryObservations
USING(sim_location, time_step_index)
JOIN
  AllCompleteInitialObservations
USING(sim_location)
ORDER BY 
  sim_location, time_step_index
)