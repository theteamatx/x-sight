PROJECT_ID=cameltrain
ZONE=us-west1-a

vm=sight-service-test
port=5540


ssh:
	gcloud compute ssh ${vm}  --project ${PROJECT_ID}     --zone ${ZONE}  -- -o ProxyCommand='corp-ssh-helper %h %p'

ssh_map_port:
	gcloud compute ssh ${vm}  --project ${PROJECT_ID}     --zone ${ZONE}  -- -o ProxyCommand='corp-ssh-helper %h %p' -NL localhost:${port}:localhost:${port}

build_calculator_worker:
	docker build --tag gcr.io/${PROJECT_ID}/test-calculator-worker:latest -f py/Dockerfile .

build_generic_worker:
	docker build --tag gcr.io/${PROJECT_ID}/test-generic-worker-local:latest -f py/Dockerfile .

run_multiple_opt_demo:
	python3 py/sight/demo/multiple_opt_demo.py --server_mode local

run_proposal_demo:
	python3 py/sight/demo/proposal_demo.py --server_mode local

run_calculator_demo:
	python3 py/sight/demo/agentic_demo/calculator_demo.py --server_mode local
