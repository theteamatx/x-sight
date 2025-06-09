CREATE OR REPLACE TABLE `cameltrain.sight_logs.{log_label}_{log_id}_{type}_simulation_unordered_time_series_log` AS (
WITH
-- AllTimeSteps AS (
--   SELECT
--     sim_location,
--     time_step,
--     time_step_index
--   FROM 
--     `cameltrain.sight_logs.{log_label}_{log_id}_value_var_log`
--   GROUP BY sim_location, time_step, time_step_index
-- ),
-- AllVarsTimeSteps AS (
-- SELECT
--   label,
--   sim_location,
--   time_step,
--   time_step_index,
-- FROM
--   AllTimeSteps
-- CROSS JOIN
--   `cameltrain.sight_logs.{log_label}_{log_id}_all_vars_log`
-- ),
-- FilledData AS (
-- SELECT
--   sim_location,
--   label,
--   IF (value_type = 'ST_STRING', string_value, 
--   IF (value_type = 'ST_DOUBLE', CAST(double_value AS STRING), 
--   IF (value_type = 'ST_INT64',  CAST(int64_value AS STRING), NULL))) AS value,
--   -- IF (double_value IS NULL, 0, double_value) AS double_value,
--   -- IF (string_value IS NULL, '', string_value) AS string_value,
--   -- IF (value_type IS NULL, 'ST_UNKNOWN', value_type) AS value_type,
--   -- double_value,
--   -- string_value,
--   time_step,
--   time_step_index,
-- FROM
--   AllVarsTimeSteps
-- LEFT JOIN
--   `cameltrain.sight_logs.{log_label}_{log_id}_value_var_log`
-- USING(sim_location, label, time_step, time_step_index)
-- -- ORDER BY time_step_index, label
-- ),
FilledData AS (
  SELECT
    sim_location,
    label,
    IF (value_type = 'ST_STRING', string_value, 
    IF (value_type = 'ST_DOUBLE', CAST(double_value AS STRING), 
    IF (value_type = 'ST_INT64',  CAST(int64_value AS STRING), NULL))) AS value,
    -- IF (double_value IS NULL, 0, double_value) AS double_value,
    -- IF (string_value IS NULL, '', string_value) AS string_value,
    -- IF (value_type IS NULL, 'ST_UNKNOWN', value_type) AS value_type,
    -- double_value,
    -- string_value,
    time_step,
    time_step_index,
  FROM
    `cameltrain.sight_logs.{log_label}_{log_id}_{type}_value_var_log`
),
States AS (
  SELECT
    sim_location,
    time_step_index,
    -- time_step_index+1 AS next_time_step_index,
    ARRAY_AGG(label ORDER BY label) AS labels,
    ARRAY_AGG(value ORDER BY label) AS values,
    -- ARRAY_AGG(double_value ORDER BY label) AS double_values,
    -- ARRAY_AGG(string_value ORDER BY label) AS string_values,
    -- ARRAY_AGG(value_type ORDER BY label) AS value_types
  FROM
    FilledData
  GROUP BY
    sim_location, time_step_index
  -- sim_location, time_step_index
)
SELECT * FROM States
)