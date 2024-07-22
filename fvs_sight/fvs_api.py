from sight.proto import sight_pb2


action_attrs = {
    "base-FERTILIZ-howManyCycle":
    sight_pb2.DecisionConfigurationStart.AttrProps(
        min_value=0,
        max_value=1,
    ),
    "base-FERTILIZ-extra_step":
    sight_pb2.DecisionConfigurationStart.AttrProps(
        min_value=0,
        max_value=1,
    ),
    "base-FERTILIZ-extra_offset":
    sight_pb2.DecisionConfigurationStart.AttrProps(
        min_value=0,
        max_value=1,
    ),
}
outcome_attrs = {
    "time_series":
    sight_pb2.DecisionConfigurationStart.AttrProps(
        min_value=0,
        max_value=1,
    ),
}
