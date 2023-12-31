# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: sight/proto/sight.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from sight.proto import example_pb2 as sight_dot_proto_dot_example__pb2
from sight.proto.widgets.pipeline.flume import flume_pb2 as sight_dot_proto_dot_widgets_dot_pipeline_dot_flume_dot_flume__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x17sight/proto/sight.proto\x12\rsight.x.proto\x1a\x19sight/proto/example.proto\x1a.sight/proto/widgets/pipeline/flume/flume.proto\"\'\n\tAttribute\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t\"\xb0\x0b\n\x06Object\x12\x10\n\x08location\x18\x01 \x01(\t\x12\r\n\x05index\x18\x02 \x01(\x03\x12\x0f\n\x07log_uid\x18\x18 \x01(\t\x12+\n\tattribute\x18\x03 \x03(\x0b\x32\x18.sight.x.proto.Attribute\x12/\n\x08sub_type\x18\x04 \x01(\x0e\x32\x1d.sight.x.proto.Object.SubType\x12#\n\x04text\x18\x05 \x01(\x0b\x32\x13.sight.x.proto.TextH\x00\x12\x30\n\x0b\x62lock_start\x18\x06 \x01(\x0b\x32\x19.sight.x.proto.BlockStartH\x00\x12,\n\tblock_end\x18\x07 \x01(\x0b\x32\x17.sight.x.proto.BlockEndH\x00\x12\x38\n\x0f\x61ttribute_start\x18\x08 \x01(\x0b\x32\x1d.sight.x.proto.AttributeStartH\x00\x12\x34\n\rattribute_end\x18\t \x01(\x0b\x32\x1b.sight.x.proto.AttributeEndH\x00\x12\x46\n\x10\x66lume_do_fn_emit\x18\x0e \x01(\x0b\x32*.sight.x.widgets.flume.proto.FlumeDoFnEmitH\x00\x12@\n\x0c\x66lume_depend\x18\x0f \x01(\x0b\x32(.sight.x.widgets.flume.proto.FlumeDependH\x00\x12%\n\x05value\x18\x10 \x01(\x0b\x32\x14.sight.x.proto.ValueH\x00\x12-\n\texception\x18\x11 \x01(\x0b\x32\x18.sight.x.proto.ExceptionH\x00\x12\'\n\x06tensor\x18\x14 \x01(\x0b\x32\x15.sight.x.proto.TensorH\x00\x12?\n\x13tensor_flow_example\x18\x15 \x01(\x0b\x32 .sight.x.proto.TensorFlowExampleH\x00\x12\x36\n\x0e\x64\x65\x63ision_point\x18\x16 \x01(\x0b\x32\x1c.sight.x.proto.DecisionPointH\x00\x12:\n\x10\x64\x65\x63ision_outcome\x18\x17 \x01(\x0b\x32\x1e.sight.x.proto.DecisionOutcomeH\x00\x12\x0c\n\x04\x66ile\x18\n \x01(\t\x12\x0c\n\x04line\x18\x0b \x01(\x05\x12\x0c\n\x04\x66unc\x18\x0c \x01(\t\x12\x1f\n\x17\x61ncestor_start_location\x18\r \x03(\t\x12.\n\x07metrics\x18\x12 \x01(\x0b\x32\x1d.sight.x.proto.Object.Metrics\x12*\n\x05order\x18\x13 \x01(\x0b\x32\x1b.sight.x.proto.Object.Order\x1aX\n\x07Metrics\x12%\n\x1dprocess_free_swap_space_bytes\x18\x01 \x01(\x03\x12&\n\x1eprocess_total_swap_space_bytes\x18\x02 \x01(\x03\x1a\x1d\n\x05Order\x12\x14\n\x0ctimestamp_ns\x18\x01 \x01(\x03\"\xae\x02\n\x07SubType\x12\x0e\n\nST_UNKNOWN\x10\x00\x12\x0b\n\x07ST_TEXT\x10\x01\x12\x12\n\x0eST_BLOCK_START\x10\x02\x12\x10\n\x0cST_BLOCK_END\x10\x03\x12\x16\n\x12ST_ATTRIBUTE_START\x10\x04\x12\x14\n\x10ST_ATTRIBUTE_END\x10\x05\x12\x17\n\x13ST_FLUME_DO_FN_EMIT\x10\x06\x12\x13\n\x0fST_FLUME_DEPEND\x10\x07\x12\x0c\n\x08ST_VALUE\x10\x08\x12\x10\n\x0cST_EXCEPTION\x10\t\x12\r\n\tST_TENSOR\x10\n\x12\x19\n\x15ST_TENSORFLOW_EXAMPLE\x10\x0c\x12\x15\n\x11ST_DECISION_POINT\x10\r\x12\x17\n\x13ST_DECISION_OUTCOME\x10\x0e\x12\n\n\x06ST_GAP\x10\x0b\x42\x12\n\x10sub_type_message\"\xec\x01\n\x12\x43onfigurationStart\x12;\n\x08sub_type\x18\x01 \x01(\x0e\x32).sight.x.proto.ConfigurationStart.SubType\x12K\n\x16\x64\x65\x63ision_configuration\x18\x02 \x01(\x0b\x32).sight.x.proto.DecisionConfigurationStartH\x00\"8\n\x07SubType\x12\x0e\n\nST_UNKNOWN\x10\x00\x12\x1d\n\x19ST_DECISION_CONFIGURATION\x10\x01\x42\x12\n\x10sub_type_message\";\n\tException\x12\x0c\n\x04type\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t\x12\x11\n\ttraceback\x18\x03 \x01(\t\"\xd7\x05\n\x06Tensor\x12/\n\x08sub_type\x18\x01 \x01(\x0e\x32\x1d.sight.x.proto.Tensor.SubType\x12\r\n\x05label\x18\x02 \x01(\t\x12\r\n\x05shape\x18\x03 \x03(\x03\x12\x11\n\tdim_label\x18\t \x03(\t\x12;\n\x0f\x64im_axis_values\x18\n \x03(\x0b\x32\".sight.x.proto.Tensor.StringValues\x12;\n\rstring_values\x18\x04 \x01(\x0b\x32\".sight.x.proto.Tensor.StringValuesH\x00\x12\x39\n\x0c\x62ytes_values\x18\x05 \x01(\x0b\x32!.sight.x.proto.Tensor.BytesValuesH\x00\x12\x39\n\x0cint64_values\x18\x06 \x01(\x0b\x32!.sight.x.proto.Tensor.Int64ValuesH\x00\x12;\n\rdouble_values\x18\x07 \x01(\x0b\x32\".sight.x.proto.Tensor.DoubleValuesH\x00\x12\x37\n\x0b\x62ool_values\x18\x08 \x01(\x0b\x32 .sight.x.proto.Tensor.BoolValuesH\x00\x1a\x1d\n\x0cStringValues\x12\r\n\x05value\x18\x01 \x03(\t\x1a\x1c\n\x0b\x42ytesValues\x12\r\n\x05value\x18\x01 \x03(\x0c\x1a\x1c\n\x0bInt64Values\x12\r\n\x05value\x18\x01 \x03(\x03\x1a\x1d\n\x0c\x44oubleValues\x12\r\n\x05value\x18\x01 \x03(\x01\x1a\x1b\n\nBoolValues\x12\r\n\x05value\x18\x01 \x03(\x08\"`\n\x07SubType\x12\x0e\n\nST_UNKNOWN\x10\x00\x12\r\n\tST_STRING\x10\x01\x12\x0c\n\x08ST_BYTES\x10\x02\x12\x0c\n\x08ST_INT64\x10\x03\x12\r\n\tST_DOUBLE\x10\x04\x12\x0b\n\x07ST_BOOL\x10\x05\x42\x0c\n\nvalue_type\"\x8e\x02\n\x11TensorFlowExample\x12/\n\rinput_example\x18\x01 \x01(\x0b\x32\x16.sight.x.proto.ExampleH\x00\x12@\n\x16input_sequence_example\x18\x02 \x01(\x0b\x32\x1e.sight.x.proto.SequenceExampleH\x00\x12\x30\n\x0eoutput_example\x18\x03 \x01(\x0b\x32\x16.sight.x.proto.ExampleH\x01\x12\x41\n\x17output_sequence_example\x18\x04 \x01(\x0b\x32\x1e.sight.x.proto.SequenceExampleH\x01\x42\x07\n\x05inputB\x08\n\x06output\")\n\x03Log\x12\"\n\x03obj\x18\x01 \x03(\x0b\x32\x15.sight.x.proto.Object\"x\n\x04Text\x12\x0c\n\x04text\x18\x01 \x01(\t\x12-\n\x08sub_type\x18\x02 \x01(\x0e\x32\x1b.sight.x.proto.Text.SubType\"3\n\x07SubType\x12\x0e\n\nST_UNKNOWN\x10\x00\x12\x0b\n\x07ST_TEXT\x10\x01\x12\x0b\n\x07ST_HTML\x10\x02\"\xbe\x02\n\x05Value\x12.\n\x08sub_type\x18\x01 \x01(\x0e\x32\x1c.sight.x.proto.Value.SubType\x12\x16\n\x0cstring_value\x18\x02 \x01(\tH\x00\x12\x15\n\x0b\x62ytes_value\x18\x03 \x01(\x0cH\x00\x12\x15\n\x0bint64_value\x18\x04 \x01(\x03H\x00\x12\x16\n\x0c\x64ouble_value\x18\x05 \x01(\x01H\x00\x12\x14\n\nbool_value\x18\x06 \x01(\x08H\x00\x12\x14\n\nnone_value\x18\x07 \x01(\x08H\x00\"m\n\x07SubType\x12\x0e\n\nST_UNKNOWN\x10\x00\x12\r\n\tST_STRING\x10\x01\x12\x0c\n\x08ST_BYTES\x10\x02\x12\x0c\n\x08ST_INT64\x10\x03\x12\r\n\tST_DOUBLE\x10\x04\x12\x0b\n\x07ST_BOOL\x10\x05\x12\x0b\n\x07ST_NONE\x10\x06\x42\x0c\n\nvalue_type\"\xa8\n\n\nBlockStart\x12\r\n\x05label\x18\x01 \x01(\t\x12\x33\n\x08sub_type\x18\x02 \x01(\x0e\x32!.sight.x.proto.BlockStart.SubType\x12J\n\x12\x66lume_do_fn_create\x18\x03 \x01(\x0b\x32,.sight.x.widgets.flume.proto.FlumeDoFnCreateH\x00\x12M\n\x14\x66lume_do_fn_start_do\x18\x04 \x01(\x0b\x32-.sight.x.widgets.flume.proto.FlumeDoFnStartDoH\x00\x12T\n\x17\x66lume_compare_fn_create\x18\x05 \x01(\x0b\x32\x31.sight.x.widgets.flume.proto.FlumeCompareFnCreateH\x00\x12\x61\n\x1e\x66lume_compare_fn_start_compare\x18\x06 \x01(\x0b\x32\x37.sight.x.widgets.flume.proto.FlumeCompareFnStartCompareH\x00\x12(\n\x04list\x18\x07 \x01(\x0b\x32\x18.sight.x.proto.ListStartH\x00\x12\\\n tensor_flow_model_training_epoch\x18\x08 \x01(\x0b\x32\x30.sight.x.proto.TensorFlowModelTrainingEpochStartH\x00\x12:\n\x10simulation_start\x18\t \x01(\x0b\x32\x1e.sight.x.proto.SimulationStartH\x00\x12O\n\x1bsimulation_parameters_start\x18\n \x01(\x0b\x32(.sight.x.proto.SimulationParametersStartH\x00\x12L\n\x1asimulation_time_step_start\x18\x0b \x01(\x0b\x32&.sight.x.proto.SimulationTimeStepStartH\x00\x12:\n\rconfiguration\x18\x0c \x01(\x0b\x32!.sight.x.proto.ConfigurationStartH\x00\"\xce\x03\n\x07SubType\x12\x0e\n\nST_UNKNOWN\x10\x00\x12\x19\n\x15ST_FLUME_DO_FN_CREATE\x10\x01\x12\x1b\n\x17ST_FLUME_DO_FN_START_DO\x10\x02\x12\x1e\n\x1aST_FLUME_COMPARE_FN_CREATE\x10\x03\x12%\n!ST_FLUME_COMPARE_FN_START_COMPARE\x10\x04\x12\x12\n\x0eST_NAMED_VALUE\x10\x05\x12\x0b\n\x07ST_LIST\x10\x06\x12\x0c\n\x08ST_TABLE\x10\x07\x12#\n\x1fST_TENSORFLOW_MODEL_APPLICATION\x10\x08\x12&\n\"ST_TENSORFLOW_MODEL_TRAINING_EPOCH\x10\t\x12 \n\x1cST_TENSORFLOW_MODEL_TRAINING\x10\n\x12\x11\n\rST_SIMULATION\x10\x0b\x12\x1c\n\x18ST_SIMULATION_PARAMETERS\x10\x0c\x12\x17\n\x13ST_SIMULATION_STATE\x10\r\x12\x1b\n\x17ST_SIMULATION_TIME_STEP\x10\x0e\x12\x19\n\x15ST_CLUSTER_ASSIGNMENT\x10\x0f\x12\x14\n\x10ST_CONFIGURATION\x10\x10\x42\x12\n\x10sub_type_message\"\xa4\x08\n\x08\x42lockEnd\x12\r\n\x05label\x18\x01 \x01(\t\x12\x31\n\x08sub_type\x18\x06 \x01(\x0e\x32\x1f.sight.x.proto.BlockEnd.SubType\x12\x1f\n\x17location_of_block_start\x18\x02 \x01(\t\x12\x1b\n\x13num_direct_contents\x18\x03 \x01(\x03\x12\x1f\n\x17num_transitive_contents\x18\x04 \x01(\x03\x12N\n\x14\x66lume_do_fn_complete\x18\x07 \x01(\x0b\x32..sight.x.widgets.flume.proto.FlumeDoFnCompleteH\x00\x12I\n\x12\x66lume_do_fn_end_do\x18\x08 \x01(\x0b\x32+.sight.x.widgets.flume.proto.FlumeDoFnEndDoH\x00\x12X\n\x19\x66lume_compare_fn_complete\x18\t \x01(\x0b\x32\x33.sight.x.widgets.flume.proto.FlumeCompareFnCompleteH\x00\x12]\n\x1c\x66lume_compare_fn_end_compare\x18\n \x01(\x0b\x32\x35.sight.x.widgets.flume.proto.FlumeCompareFnEndCompareH\x00\x12\x30\n\x07metrics\x18\x0c \x01(\x0b\x32\x1f.sight.x.proto.BlockEnd.Metrics\x1a\"\n\x07Metrics\x12\x17\n\x0f\x65lapsed_time_ns\x18\x01 \x01(\x03\"\xb8\x03\n\x07SubType\x12\x0e\n\nST_UNKNOWN\x10\x00\x12\x1b\n\x17ST_FLUME_DO_FN_COMPLETE\x10\x01\x12\x19\n\x15ST_FLUME_DO_FN_END_DO\x10\x02\x12 \n\x1cST_FLUME_COMPARE_FN_COMPLETE\x10\x03\x12#\n\x1fST_FLUME_COMPARE_FN_END_COMPARE\x10\x04\x12\x12\n\x0eST_NAMED_VALUE\x10\x05\x12\x0b\n\x07ST_LIST\x10\x06\x12\x0c\n\x08ST_TABLE\x10\x07\x12#\n\x1fST_TENSORFLOW_MODEL_APPLICATION\x10\x08\x12&\n\"ST_TENSORFLOW_MODEL_TRAINING_EPOCH\x10\t\x12 \n\x1cST_TENSORFLOW_MODEL_TRAINING\x10\n\x12\x11\n\rST_SIMULATION\x10\x0b\x12\x1c\n\x18ST_SIMULATION_PARAMETERS\x10\x0c\x12\x17\n\x13ST_SIMULATION_STATE\x10\r\x12\x1b\n\x17ST_SIMULATION_TIME_STEP\x10\x0e\x12\x19\n\x15ST_CLUSTER_ASSIGNMENT\x10\x0f\x42\x12\n\x10sub_type_message\"\xaf\x01\n\tListStart\x12\x32\n\x08sub_type\x18\x01 \x01(\x0e\x32 .sight.x.proto.ListStart.SubType\"n\n\x07SubType\x12\x0e\n\nST_UNKNOWN\x10\x00\x12\x12\n\x0eST_HOMOGENEOUS\x10\x01\x12\x14\n\x10ST_HETEROGENEOUS\x10\x02\x12\n\n\x06ST_MAP\x10\x03\x12\x10\n\x0cST_MAP_ENTRY\x10\x04\x12\x0b\n\x07ST_DICT\x10\x05\"J\n!TensorFlowModelTrainingEpochStart\x12\x11\n\tepoch_num\x18\x01 \x01(\x03\x12\x12\n\nbatch_size\x18\x02 \x01(\x03\"=\n\x0e\x41ttributeStart\x12+\n\tattribute\x18\x01 \x01(\x0b\x32\x18.sight.x.proto.Attribute\"\x1b\n\x0c\x41ttributeEnd\x12\x0b\n\x03key\x18\x01 \x01(\t\"\xb2\x03\n\x06Params\x12\r\n\x05local\x18\x01 \x01(\x08\x12\x14\n\x0clog_dir_path\x18\x02 \x01(\t\x12\r\n\x05label\x18\x03 \x01(\t\x12\x13\n\x0btext_output\x18\x04 \x01(\x08\x12\x17\n\x0f\x63olumnio_output\x18\x05 \x01(\x08\x12\x18\n\x10\x63\x61pacitor_output\x18\x06 \x01(\x08\x12\x11\n\tlog_owner\x18\x07 \x01(\t\x12\x13\n\x0bpath_prefix\x18\x08 \x01(\t\x12\x1a\n\x12\x63ontainer_location\x18\t \x01(\t\x12\n\n\x02id\x18\n \x01(\x03\x12\x15\n\rsilent_logger\x18\x0b \x01(\x08\x12\x11\n\tin_memory\x18\x0c \x01(\x08\x12\x13\n\x0b\x61vro_output\x18\r \x01(\x08\x12\x12\n\nproject_id\x18\x0e \x01(\t\x12\x13\n\x0b\x62ucket_name\x18\x0f \x01(\t\x12\x10\n\x08gcp_path\x18\x10 \x01(\t\x12\x13\n\x0b\x66ile_format\x18\x11 \x01(\t\x12\x14\n\x0c\x64\x61taset_name\x18\x12 \x01(\t\x12\x1c\n\x14\x65xternal_file_format\x18\x13 \x01(\t\x12\x19\n\x11\x65xternal_file_uri\x18\x14 \x01(\t\"\x11\n\x0fSimulationStart\"\x1b\n\x19SimulationParametersStart\"\xb8\x02\n\x17SimulationTimeStepStart\x12\x17\n\x0ftime_step_index\x18\x01 \x03(\x03\x12\x11\n\ttime_step\x18\x02 \x01(\x02\x12\x16\n\x0etime_step_size\x18\x03 \x01(\x02\x12M\n\x0ftime_step_units\x18\x04 \x01(\x0e\x32\x34.sight.x.proto.SimulationTimeStepStart.TimeStepUnits\"\x89\x01\n\rTimeStepUnits\x12\x0f\n\x0bTSU_UNKNOWN\x10\x00\x12\x0e\n\nTSU_SECOND\x10\x01\x12\x0e\n\nTSU_MINUTE\x10\x02\x12\x0c\n\x08TSU_HOUR\x10\x03\x12\x0b\n\x07TSU_DAY\x10\x04\x12\r\n\tTSU_MONTH\x10\x05\x12\x0f\n\x0bTSU_QUARTER\x10\x06\x12\x0c\n\x08TSU_YEAR\x10\x07\"\xe0\x0e\n\x1a\x44\x65\x63isionConfigurationStart\x12O\n\x0eoptimizer_type\x18\x04 \x01(\x0e\x32\x37.sight.x.proto.DecisionConfigurationStart.OptimizerType\x12R\n\rchoice_config\x18\x01 \x03(\x0b\x32;.sight.x.proto.DecisionConfigurationStart.ChoiceConfigEntry\x12N\n\x0bstate_attrs\x18\x02 \x03(\x0b\x32\x39.sight.x.proto.DecisionConfigurationStart.StateAttrsEntry\x12P\n\x0c\x61\x63tion_attrs\x18\x03 \x03(\x0b\x32:.sight.x.proto.DecisionConfigurationStart.ActionAttrsEntry\x1a\x0e\n\x0cVizierConfig\x1a\xbf\x01\n\nAcmeConfig\x12\x10\n\x08\x65nv_name\x18\x01 \x01(\t\x12\x11\n\tstate_min\x18\x02 \x03(\x02\x12\x11\n\tstate_max\x18\x03 \x03(\x02\x12\x1a\n\x12state_param_length\x18\x04 \x01(\x03\x12\x12\n\naction_min\x18\x05 \x03(\x02\x12\x12\n\naction_max\x18\x06 \x03(\x02\x12\x1b\n\x13\x61\x63tion_param_length\x18\x07 \x01(\x03\x12\x18\n\x10possible_actions\x18\x08 \x01(\x03\x1a\x35\n\x16GeneticAlgorithmConfig\x12\x1b\n\x13max_population_size\x18\x01 \x01(\x03\x1a\x18\n\x16\x45xhaustiveSearchConfig\x1a\xce\x01\n\tLLMConfig\x12S\n\talgorithm\x18\x01 \x01(\x0e\x32@.sight.x.proto.DecisionConfigurationStart.LLMConfig.LLMAlgorithm\x12\x13\n\x0b\x64\x65scription\x18\x02 \x01(\t\"W\n\x0cLLMAlgorithm\x12\x0e\n\nLA_UNKNOWN\x10\x00\x12\x11\n\rLA_TEXT_BISON\x10\x01\x12\x11\n\rLA_CHAT_BISON\x10\x02\x12\x11\n\rLA_GEMINI_PRO\x10\x03\x1a\xd4\x03\n\x0c\x43hoiceConfig\x12O\n\rvizier_config\x18\x01 \x01(\x0b\x32\x36.sight.x.proto.DecisionConfigurationStart.VizierConfigH\x00\x12K\n\x0b\x61\x63me_config\x18\x02 \x01(\x0b\x32\x34.sight.x.proto.DecisionConfigurationStart.AcmeConfigH\x00\x12\x64\n\x18genetic_algorithm_config\x18\x03 \x01(\x0b\x32@.sight.x.proto.DecisionConfigurationStart.GeneticAlgorithmConfigH\x00\x12\x64\n\x18\x65xhaustive_search_config\x18\x04 \x01(\x0b\x32@.sight.x.proto.DecisionConfigurationStart.ExhaustiveSearchConfigH\x00\x12I\n\nllm_config\x18\x05 \x01(\x0b\x32\x33.sight.x.proto.DecisionConfigurationStart.LLMConfigH\x00\x42\x0f\n\rchoice_config\x1ak\n\x11\x43hoiceConfigEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\x45\n\x05value\x18\x02 \x01(\x0b\x32\x36.sight.x.proto.DecisionConfigurationStart.ChoiceConfig:\x02\x38\x01\x1au\n\tAttrProps\x12\x11\n\tmin_value\x18\x01 \x01(\x02\x12\x11\n\tmax_value\x18\x02 \x01(\x02\x12\x11\n\tstep_size\x18\x03 \x01(\x02\x12\x1a\n\x12valid_float_values\x18\x04 \x03(\x02\x12\x13\n\x0b\x64\x65scription\x18\x05 \x01(\t\x1a\x66\n\x0fStateAttrsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\x42\n\x05value\x18\x02 \x01(\x0b\x32\x33.sight.x.proto.DecisionConfigurationStart.AttrProps:\x02\x38\x01\x1ag\n\x10\x41\x63tionAttrsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\x42\n\x05value\x18\x02 \x01(\x0b\x32\x33.sight.x.proto.DecisionConfigurationStart.AttrProps:\x02\x38\x01\"{\n\rOptimizerType\x12\x0e\n\nOT_UNKNOWN\x10\x00\x12\r\n\tOT_VIZIER\x10\x01\x12\x0b\n\x07OT_ACME\x10\x02\x12\x18\n\x14OT_GENETIC_ALGORITHM\x10\x03\x12\x18\n\x14OT_EXHAUSTIVE_SEARCH\x10\x04\x12\n\n\x06OT_LLM\x10\x05\"A\n\rDecisionParam\x12\x0b\n\x03key\x18\x01 \x01(\t\x12#\n\x05value\x18\x02 \x01(\x0b\x32\x14.sight.x.proto.Value\"\xa5\x01\n\rDecisionPoint\x12\x14\n\x0c\x63hoice_label\x18\x01 \x01(\t\x12\x15\n\rchosen_option\x18\x02 \x01(\t\x12\x33\n\rchoice_params\x18\x03 \x03(\x0b\x32\x1c.sight.x.proto.DecisionParam\x12\x32\n\x0cstate_params\x18\x04 \x03(\x0b\x32\x1c.sight.x.proto.DecisionParam\"Q\n\x0f\x44\x65\x63isionOutcome\x12\x15\n\routcome_label\x18\x01 \x01(\t\x12\x15\n\routcome_value\x18\x02 \x01(\x02\x12\x10\n\x08\x64iscount\x18\x03 \x01(\x02\x62\x06proto3')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'sight.proto.sight_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _DECISIONCONFIGURATIONSTART_CHOICECONFIGENTRY._options = None
  _DECISIONCONFIGURATIONSTART_CHOICECONFIGENTRY._serialized_options = b'8\001'
  _DECISIONCONFIGURATIONSTART_STATEATTRSENTRY._options = None
  _DECISIONCONFIGURATIONSTART_STATEATTRSENTRY._serialized_options = b'8\001'
  _DECISIONCONFIGURATIONSTART_ACTIONATTRSENTRY._options = None
  _DECISIONCONFIGURATIONSTART_ACTIONATTRSENTRY._serialized_options = b'8\001'
  _ATTRIBUTE._serialized_start=117
  _ATTRIBUTE._serialized_end=156
  _OBJECT._serialized_start=159
  _OBJECT._serialized_end=1615
  _OBJECT_METRICS._serialized_start=1171
  _OBJECT_METRICS._serialized_end=1259
  _OBJECT_ORDER._serialized_start=1261
  _OBJECT_ORDER._serialized_end=1290
  _OBJECT_SUBTYPE._serialized_start=1293
  _OBJECT_SUBTYPE._serialized_end=1595
  _CONFIGURATIONSTART._serialized_start=1618
  _CONFIGURATIONSTART._serialized_end=1854
  _CONFIGURATIONSTART_SUBTYPE._serialized_start=1778
  _CONFIGURATIONSTART_SUBTYPE._serialized_end=1834
  _EXCEPTION._serialized_start=1856
  _EXCEPTION._serialized_end=1915
  _TENSOR._serialized_start=1918
  _TENSOR._serialized_end=2645
  _TENSOR_STRINGVALUES._serialized_start=2384
  _TENSOR_STRINGVALUES._serialized_end=2413
  _TENSOR_BYTESVALUES._serialized_start=2415
  _TENSOR_BYTESVALUES._serialized_end=2443
  _TENSOR_INT64VALUES._serialized_start=2445
  _TENSOR_INT64VALUES._serialized_end=2473
  _TENSOR_DOUBLEVALUES._serialized_start=2475
  _TENSOR_DOUBLEVALUES._serialized_end=2504
  _TENSOR_BOOLVALUES._serialized_start=2506
  _TENSOR_BOOLVALUES._serialized_end=2533
  _TENSOR_SUBTYPE._serialized_start=2535
  _TENSOR_SUBTYPE._serialized_end=2631
  _TENSORFLOWEXAMPLE._serialized_start=2648
  _TENSORFLOWEXAMPLE._serialized_end=2918
  _LOG._serialized_start=2920
  _LOG._serialized_end=2961
  _TEXT._serialized_start=2963
  _TEXT._serialized_end=3083
  _TEXT_SUBTYPE._serialized_start=3032
  _TEXT_SUBTYPE._serialized_end=3083
  _VALUE._serialized_start=3086
  _VALUE._serialized_end=3404
  _VALUE_SUBTYPE._serialized_start=3281
  _VALUE_SUBTYPE._serialized_end=3390
  _BLOCKSTART._serialized_start=3407
  _BLOCKSTART._serialized_end=4727
  _BLOCKSTART_SUBTYPE._serialized_start=4245
  _BLOCKSTART_SUBTYPE._serialized_end=4707
  _BLOCKEND._serialized_start=4730
  _BLOCKEND._serialized_end=5790
  _BLOCKEND_METRICS._serialized_start=5293
  _BLOCKEND_METRICS._serialized_end=5327
  _BLOCKEND_SUBTYPE._serialized_start=5330
  _BLOCKEND_SUBTYPE._serialized_end=5770
  _LISTSTART._serialized_start=5793
  _LISTSTART._serialized_end=5968
  _LISTSTART_SUBTYPE._serialized_start=5858
  _LISTSTART_SUBTYPE._serialized_end=5968
  _TENSORFLOWMODELTRAININGEPOCHSTART._serialized_start=5970
  _TENSORFLOWMODELTRAININGEPOCHSTART._serialized_end=6044
  _ATTRIBUTESTART._serialized_start=6046
  _ATTRIBUTESTART._serialized_end=6107
  _ATTRIBUTEEND._serialized_start=6109
  _ATTRIBUTEEND._serialized_end=6136
  _PARAMS._serialized_start=6139
  _PARAMS._serialized_end=6573
  _SIMULATIONSTART._serialized_start=6575
  _SIMULATIONSTART._serialized_end=6592
  _SIMULATIONPARAMETERSSTART._serialized_start=6594
  _SIMULATIONPARAMETERSSTART._serialized_end=6621
  _SIMULATIONTIMESTEPSTART._serialized_start=6624
  _SIMULATIONTIMESTEPSTART._serialized_end=6936
  _SIMULATIONTIMESTEPSTART_TIMESTEPUNITS._serialized_start=6799
  _SIMULATIONTIMESTEPSTART_TIMESTEPUNITS._serialized_end=6936
  _DECISIONCONFIGURATIONSTART._serialized_start=6939
  _DECISIONCONFIGURATIONSTART._serialized_end=8827
  _DECISIONCONFIGURATIONSTART_VIZIERCONFIG._serialized_start=7296
  _DECISIONCONFIGURATIONSTART_VIZIERCONFIG._serialized_end=7310
  _DECISIONCONFIGURATIONSTART_ACMECONFIG._serialized_start=7313
  _DECISIONCONFIGURATIONSTART_ACMECONFIG._serialized_end=7504
  _DECISIONCONFIGURATIONSTART_GENETICALGORITHMCONFIG._serialized_start=7506
  _DECISIONCONFIGURATIONSTART_GENETICALGORITHMCONFIG._serialized_end=7559
  _DECISIONCONFIGURATIONSTART_EXHAUSTIVESEARCHCONFIG._serialized_start=7561
  _DECISIONCONFIGURATIONSTART_EXHAUSTIVESEARCHCONFIG._serialized_end=7585
  _DECISIONCONFIGURATIONSTART_LLMCONFIG._serialized_start=7588
  _DECISIONCONFIGURATIONSTART_LLMCONFIG._serialized_end=7794
  _DECISIONCONFIGURATIONSTART_LLMCONFIG_LLMALGORITHM._serialized_start=7707
  _DECISIONCONFIGURATIONSTART_LLMCONFIG_LLMALGORITHM._serialized_end=7794
  _DECISIONCONFIGURATIONSTART_CHOICECONFIG._serialized_start=7797
  _DECISIONCONFIGURATIONSTART_CHOICECONFIG._serialized_end=8265
  _DECISIONCONFIGURATIONSTART_CHOICECONFIGENTRY._serialized_start=8267
  _DECISIONCONFIGURATIONSTART_CHOICECONFIGENTRY._serialized_end=8374
  _DECISIONCONFIGURATIONSTART_ATTRPROPS._serialized_start=8376
  _DECISIONCONFIGURATIONSTART_ATTRPROPS._serialized_end=8493
  _DECISIONCONFIGURATIONSTART_STATEATTRSENTRY._serialized_start=8495
  _DECISIONCONFIGURATIONSTART_STATEATTRSENTRY._serialized_end=8597
  _DECISIONCONFIGURATIONSTART_ACTIONATTRSENTRY._serialized_start=8599
  _DECISIONCONFIGURATIONSTART_ACTIONATTRSENTRY._serialized_end=8702
  _DECISIONCONFIGURATIONSTART_OPTIMIZERTYPE._serialized_start=8704
  _DECISIONCONFIGURATIONSTART_OPTIMIZERTYPE._serialized_end=8827
  _DECISIONPARAM._serialized_start=8829
  _DECISIONPARAM._serialized_end=8894
  _DECISIONPOINT._serialized_start=8897
  _DECISIONPOINT._serialized_end=9062
  _DECISIONOUTCOME._serialized_start=9064
  _DECISIONOUTCOME._serialized_end=9145
# @@protoc_insertion_point(module_scope)
