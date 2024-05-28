AllTimeSteps AS (
  SELECT
    sim_location,
    time_step,
    time_step_index
  FROM Data
  GROUP BY sim_location, time_step, time_step_index
),
AllVarsTimeSteps AS (
SELECT
  label,
  sim_location,
  time_step,
  time_step_index,
FROM
  AllTimeSteps
CROSS JOIN
  AllVars
),
FilledData AS (
SELECT
  sim_location,
  label,
  IF (double_value IS NULL, 0, double_value) AS double_value,
  time_step,
  time_step_index,
FROM
  AllVarsTimeSteps
LEFT JOIN
  Data
USING(sim_location, label, time_step, time_step_index)
ORDER BY time_step_index, label
),
States AS (
SELECT
  sim_location,
  time_step_index,
  time_step_index+1 AS next_time_step_index,
  ARRAY_AGG(double_value) AS values
FROM
  FilledData
GROUP BY
  sim_location, time_step_index
ORDER BY sim_location, time_step_index
)
SELECT * FROM States