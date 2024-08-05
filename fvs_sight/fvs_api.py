from sight.proto import sight_pb2


action_attrs = {
    "a1":
    sight_pb2.DecisionConfigurationStart.AttrProps(
        min_value=0,
        max_value=1,
    ),
    "a2":
    sight_pb2.DecisionConfigurationStart.AttrProps(
        min_value=0,
        max_value=1,
    ),
    "a3":
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
