PROJECT_ID=cameltrain
ZONE=us-west1-a

vm=sight-service-test
port=5540


ssh:
	gcloud compute ssh ${vm}  --project ${PROJECT_ID}     --zone ${ZONE}  -- -o ProxyCommand='corp-ssh-helper %h %p'

ssh_map_port:
	gcloud compute ssh ${vm}  --project ${PROJECT_ID}     --zone ${ZONE}  -- -o ProxyCommand='corp-ssh-helper %h %p' -NL localhost:${port}:localhost:${port}

test-all:
	python py/tests/discover_and_run_tests.py

proto-sight:
	protoc -I="py/" --python_out="py/" sight/proto/sight.proto
proto-server:
	python -m grpc_tools.protoc --include_imports --include_source_info --proto_path=${GOOGLEAPIS_DIR} --proto_path="py/" --proto_path=. --python_out=. --grpc_python_out=. --descriptor_set_out=sight_service/proto/api_descriptor.pb sight_service/proto/service.proto

build_calculator_worker:
	docker build --tag gcr.io/${PROJECT_ID}/test-calculator-worker:latest -f py/Dockerfile .

build_pyrolyzer_worker:
	docker build --tag gcr.io/${PROJECT_ID}/test-pyrolyzer-worker:latest -f py/Dockerfile .

build_generic_worker:
	docker build --tag gcr.io/${PROJECT_ID}/test-generic-worker-local:latest -f py/Dockerfile .

run_multiple_opt_demo:
	python3 py/sight/demo/multiple_opt_demo.py --server_mode local

run_proposal_demo:
	python3 py/sight/demo/proposal_demo.py --server_mode local --cache_mode gcs

run_calculator_demo:
	python3 py/sight/demo/agentic_demo/calculator_demo.py --server_mode local


run_local_server:
	python sight_service/service_root.py

build_and_push_sight_server:
	docker build --tag gcr.io/$PROJECT_ID/sight-default:$(git rev-parse --abbrev-ref HEAD)-$(git rev-parse --short HEAD) -f sight_service/Dockerfile . &&     gcloud auth print-access-token | docker login -u oauth2accesstoken --password-stdin https://gcr.io &&     docker push gcr.io/$PROJECT_ID/sight-default:$(git rev-parse --abbrev-ref HEAD)-$(git rev-parse --short HEAD)
