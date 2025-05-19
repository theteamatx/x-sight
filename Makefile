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
