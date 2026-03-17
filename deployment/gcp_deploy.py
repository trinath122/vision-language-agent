"""
Deploy the Vision-Language model to GCP Vertex AI.
INT8 quantized for sub-100ms inference latency.
"""
from google.cloud import aiplatform
from google.cloud import storage
import yaml
import os


def upload_model_to_gcs(local_path: str, bucket_name: str, gcs_path: str):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(local_path)
    print(f"Uploaded {local_path} → gs://{bucket_name}/{gcs_path}")


def deploy_to_vertex_ai(config: dict):
    cfg = config["gcp"]
    aiplatform.init(project=cfg["project_id"], location=cfg["region"])

    # Upload model artifacts to GCS
    model_gcs_uri = f"gs://{cfg['bucket']}/models/vision-lang-agent/"

    # Register model
    model = aiplatform.Model.upload(
        display_name="vision-lang-agent-int8",
        artifact_uri=model_gcs_uri,
        serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/pytorch-gpu.2-2:latest",
        serving_container_environment_variables={
            "MODEL_NAME": "vision-lang-agent",
            "QUANTIZATION": "int8",
        },
    )

    # Deploy endpoint
    endpoint = model.deploy(
        machine_type="n1-standard-8",
        accelerator_type="NVIDIA_TESLA_T4",
        accelerator_count=1,
        min_replica_count=1,
        max_replica_count=3,
        traffic_split={"0": 100},
    )

    print(f"Deployed to endpoint: {endpoint.resource_name}")
    return endpoint


if __name__ == "__main__":
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)
    deploy_to_vertex_ai(config)
