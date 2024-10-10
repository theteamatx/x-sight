CREATE OR REPLACE TABLE `cameltrain.sight_logs.{log_id}_{type}_all_vars_log` AS (
  SELECT 
    label 
  FROM 
    `cameltrain.sight_logs.{log_id}_{type}_value_var_log`
  GROUP BY label 
  ORDER BY label
)