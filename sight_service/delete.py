from sight_service.normalizer import Normalizer

normalizer_obj = Normalizer()


action_attrs={}
for i in range(num_attributes):
  key = f"{i}"  # Generate unique keys
  action_attrs[key] = sight_pb2.DecisionConfigurationStart.AttrProps(
      min_value=attr_range[0],
      max_value=attr_range[1],
  )
