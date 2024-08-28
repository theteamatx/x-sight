CREATE OR REPLACE TABLE `cameltrain.sight_logs.{log_id}_simulation_ordered_time_series_log` AS (
SELECT
  sim_location,
  time_step_index,
  time_step_index+1 AS next_time_step_index,
  double_values,
  string_values,
  value_types
FROM
  `cameltrain.sight_logs.{log_id}_simulation_unordered_time_series_log`
ORDER BY sim_location, time_step_index
)