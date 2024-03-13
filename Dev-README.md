# Sight Developer Documentation

This document contains some commands, useful while developing the Sight system.


## generating python client for proto files

after any modifications in the proto files, you need to generate it _pb2 client.

Run from x-sight/py folder :
```
protoc -I=. --python_out=. sight/proto/*.proto
protoc -I=. --python_out=. sight/proto/widgets/pipeline/flume/*.proto
```

Run from x-sight folder :
```
protoc -I=. --python_out=. sight_service/proto/numproto/protobuf/ndarray.proto
```
