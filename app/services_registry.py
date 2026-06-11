"""
Custom service registry for ADK.

Registers Firestore session storage under the `firestore://` URI scheme.
"""

from __future__ import annotations

import os
from urllib.parse import urlparse

from google.adk.cli.service_registry import get_service_registry
from google.adk.integrations.firestore.firestore_session_service import (
    FirestoreSessionService,
)


def _sanitize_state(state: dict | None) -> dict | None:
    if not state:
        return state
    return {k: v for k, v in state.items() if not k.startswith("__")}


class SafeFirestoreSessionService(FirestoreSessionService):
    async def create_session(
        self,
        *,
        app_name: str,
        user_id: str,
        state: dict | None = None,
        session_id: str | None = None,
    ):
        safe_state = _sanitize_state(state)
        return await super().create_session(
            app_name=app_name,
            user_id=user_id,
            state=safe_state,
            session_id=session_id,
        )


def _firestore_session_factory(uri: str, **_kwargs):
    parsed = urlparse(uri)
    root_collection = os.environ.get("ADK_FIRESTORE_ROOT_COLLECTION")
    if parsed.path and parsed.path not in ("", "/"):
        root_collection = parsed.path.lstrip("/")
    return SafeFirestoreSessionService(root_collection=root_collection)


get_service_registry().register_session_service("firestore", _firestore_session_factory)
