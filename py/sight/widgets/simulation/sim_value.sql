CREATE OR REPLACE TABLE `cameltrain.sight_logs.{log_id}_{type}_value_var_log` AS (
WITH
Values AS (
  SELECT
    location,
    sub_type,
    value.double_value,
    value.string_value,
    value.int64_value,
    value.sub_type AS value_type,
    ancestor_start_location,
    array_reverse(ancestor_start_location)[offset(1)] AS parent_loc
  FROM `cameltrain.sight_logs.{log_id}_log`
  WHERE sub_type='ST_VALUE'
),
Data AS (
  SELECT
    sim_location,
    named_var_location,
    label,
    double_value,
    string_value,
    int64_value,
    value_type,
    time_step,
    -- ARRAY_TO_STRING(time_step_index) AS time_step_index,
    time_step_index,
    time_step_size,
    time_step_units,
  FROM
    Values
  JOIN
    `cameltrain.sight_logs.{log_id}_{type}_named_var_log` AS NamedVar
  ON NamedVar.named_var_location = Values.parent_loc
)
SELECT * FROM Data
)