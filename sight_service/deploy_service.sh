#!/usr/bin/bash
# Builds the docker container for the Sight service 
docker build --tag gcr.io/$PROJECT_ID/sight-$1 -f sight_service/Dockerfile .
docker push gcr.io/$PROJECT_ID/sight-$1
gcloud run deploy sight-$1 --image=gcr.io/$PROJECT_ID/sight-$1:latest --allow-unauthenticated --service-account=sight-service-account@$PROJECT_ID.iam.gserviceaccount.com --concurrency=default --cpu=2 --memory=8Gi --min-instances=1 --max-instances=1 --no-cpu-throttling --region=us-central1 --project=$PROJECT_ID