"""
S3 document fetcher for Iqidis documents.
Downloads from iqidis-artifact bucket using artifact.storage_key.
"""
import os
from typing import Optional

S3_BUCKET = "iqidis-artifact"


def download_from_s3(storage_key: str, bucket: str = S3_BUCKET) -> Optional[bytes]:
    """Download document bytes from S3."""
    try:
        import boto3
        from botocore.exceptions import ClientError
    except ImportError:
        raise ImportError("boto3 required. pip install boto3")

    region = os.getenv("AWS_REGION", "us-east-1")
    client = boto3.client(
        "s3",
        region_name=region,
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )
    try:
        response = client.get_object(Bucket=bucket, Key=storage_key)
        return response["Body"].read()
    except ClientError as e:
        print(f"S3 download failed for {storage_key}: {e}")
        return None
