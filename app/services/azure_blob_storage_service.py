"""
services/azure_blob_storage_service.py
──────────────────────────────────────
Azure Blob Storage service for uploading videos and schedule JSON.
"""

from __future__ import annotations

import io
import json
import os
import uuid
from typing import Any

import requests
from azure.storage.blob import BlobServiceClient, ContentSettings


class AzureBlobStorageService:
    def __init__(self, conn_string: str, container: str):
        self.connection_string = conn_string
        self.container_name = container

    def upload_blob_from_url(self, file_url: str) -> str:
        blob_service_client = BlobServiceClient.from_connection_string(self.connection_string)

        response = requests.get(file_url, stream=True, timeout=30)
        response.raise_for_status()
        blob_data = response.content
        content_type = response.headers.get("Content-Type", "application/octet-stream")

        blob_name = f"{uuid.uuid4()}-{os.path.basename(file_url).split('?')[0]}"

        container_client = blob_service_client.get_container_client(self.container_name)
        content_settings = ContentSettings(content_type=content_type)
        blob_client = container_client.get_blob_client(blob_name)
        blob_client.upload_blob(blob_data, blob_type="BlockBlob", content_settings=content_settings)

        print(f"[Blob] Uploaded from URL: {blob_client.url}")
        return blob_client.url

    def upload_blob_from_disk(self, file_path: str, blob_name: str | None = None) -> str:
        blob_service_client = BlobServiceClient.from_connection_string(self.connection_string)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()
        mime_types = {
            ".mp4": "video/mp4",
            ".mov": "video/quicktime",
            ".jpeg": "image/jpeg",
            ".jpg": "image/jpeg",
            ".png": "image/png",
            ".wav": "audio/wav",
        }
        content_type = mime_types.get(ext, "application/octet-stream")

        safe_blob_name = blob_name.lstrip("/") if blob_name else None
        blob_name = safe_blob_name or f"{uuid.uuid4()}-{os.path.basename(file_path)}"

        with open(file_path, "rb") as fh:
            blob_data = fh.read()

        container_client = blob_service_client.get_container_client(self.container_name)
        content_settings = ContentSettings(content_type=content_type)
        blob_client = container_client.get_blob_client(blob_name)
        blob_client.upload_blob(
            io.BytesIO(blob_data),
            blob_type="BlockBlob",
            content_settings=content_settings,
        )

        print(f"[Blob] Uploaded from disk: {blob_client.url}")
        return blob_client.url

    def download_json_from_url(self, blob_url: str, account_key: str) -> Any:
        from urllib.parse import urlparse

        parsed_url = urlparse(blob_url)
        storage_account_url = f"https://{parsed_url.netloc}"
        path_parts = parsed_url.path.lstrip("/").split("/")
        container_name = path_parts[0]
        blob_name = "/".join(path_parts[1:])

        blob_service_client = BlobServiceClient(account_url=storage_account_url, credential=account_key)
        blob_client = blob_service_client.get_container_client(container_name).get_blob_client(blob_name)

        if not blob_client.exists():
            raise FileNotFoundError(f"Blob '{blob_name}' not found in '{container_name}'")

        return json.loads(blob_client.download_blob().readall())

    def upload_json_to_url(self, data: Any, blob_url: str, account_key: str) -> None:
        from urllib.parse import urlparse

        parsed_url = urlparse(blob_url)
        storage_account_url = f"https://{parsed_url.netloc}"
        path_parts = parsed_url.path.lstrip("/").split("/")
        container_name = path_parts[0]
        blob_name = "/".join(path_parts[1:])

        blob_service_client = BlobServiceClient(account_url=storage_account_url, credential=account_key)
        blob_client = blob_service_client.get_container_client(container_name).get_blob_client(blob_name)
        blob_client.upload_blob(json.dumps(data), overwrite=True)
