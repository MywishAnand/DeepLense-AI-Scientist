"""
REST API utilities for MCP-wrapped tools.

Usage (recommended):
- Load env vars once in your MCP server entrypoint (e.g., mcp_server.py):
    from dotenv import load_dotenv
    load_dotenv()

- Then import and use these helpers in tool modules.

This file provides:
- get_api_key(): strict env lookup with optional fallback
- RestClient: requests.Session + retries + timeouts + JSON handling
- make_api_request(): thin convenience wrapper around the default client
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Mapping

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# Logging
logger = logging.getLogger(__name__)


# Configuration for the REST client.
@dataclass(frozen=True)
class RestClientConfig:
    """Configuration for REST client behavior."""

    timeout_s: int = 10
    retries_total: int = 3
    retries_backoff_factor: float = 0.5
    retry_statuses: tuple[int, ...] = (500, 502, 503, 504)
    retry_methods: tuple[str, ...] = ("GET", "POST")
    user_agent: str = "mcp-rest-client/1.0"


# Helper function to get the API key from the environment variables.
def get_api_key(key_name: str, fallback_key: Optional[str] = None) -> str:
    """
    Fetch an API key from environment variables.

    Args:
        key_name: Environment variable name (e.g. "NASA_API_KEY")
        fallback_key: Optional fallback value (use sparingly; prefer raising)

    Returns:
        API key as string

    Raises:
        EnvironmentError if missing and no fallback is provided.
    """
    api_key = os.getenv(key_name)
    if api_key:
        return api_key

    if fallback_key is not None:
        logger.warning(
            "Environment variable %s is not set; using fallback key.", key_name
        )
        return fallback_key

    raise EnvironmentError(f"Missing required environment variable: {key_name}")


# REST client class.
class RestClient:
    """
    A small, reusable REST client intended for MCP tool backends.

    Features:
    - requests.Session connection pooling
    - retries on transient 5xx errors
    - consistent JSON response handling
    - structured logging (no print spam)
    """

    def __init__(self, config: RestClientConfig | None = None) -> None:
        self.config = config or RestClientConfig()
        self.session = self._create_session(self.config)

    @staticmethod
    def _create_session(config: RestClientConfig) -> requests.Session:
        session = requests.Session()
        session.headers.update({"User-Agent": config.user_agent})

        retry = Retry(
            total=config.retries_total,
            backoff_factor=config.retries_backoff_factor,
            status_forcelist=list(config.retry_statuses),
            allowed_methods=set(m.upper() for m in config.retry_methods),
            raise_on_status=False,
            respect_retry_after_header=True,
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session

    def _handle_error(
        self, method: str, url: str, params_or_body: Any, e: Exception, resp: Any = None
    ) -> None:
        """Shared error logging for GET and POST."""
        if isinstance(e, requests.Timeout):
            logger.error("Timeout | %s %s | %s", method, url, params_or_body)
        elif isinstance(e, requests.ConnectionError):
            logger.error("Connection error | %s %s | err=%s", method, url, e)
        elif isinstance(e, requests.HTTPError):
            status = getattr(resp, "status_code", "unknown")
            body_snippet = ""
            try:
                body_snippet = (resp.text or "")[:300]
            except Exception:
                pass
            logger.error(
                "HTTP error %s | %s %s | %s | err=%s | body=%s",
                status,
                method,
                url,
                params_or_body,
                e,
                body_snippet,
            )
        elif isinstance(e, ValueError):
            logger.error("Invalid JSON response | %s %s | err=%s", method, url, e)
        elif isinstance(e, requests.RequestException):
            logger.error("Request exception | %s %s | err=%s", method, url, e)
        else:
            logger.exception("Unexpected error | %s %s | err=%s", method, url, e)

    def get_json(
        self,
        url: str,
        params: Mapping[str, Any] | None = None,
        *,
        timeout_s: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Perform a GET request and parse JSON.

        Returns:
            dict JSON on success, else None.
        """
        effective_timeout = (
            timeout_s if timeout_s is not None else self.config.timeout_s
        )
        resp = None
        try:
            resp = self.session.get(
                url, params=dict(params or {}), timeout=effective_timeout
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            self._handle_error("GET", url, params, e, resp)
            return None

    def post_json(
        self,
        url: str,
        json_body: Dict[str, Any] | None = None,
        *,
        headers: Dict[str, str] | None = None,
        timeout_s: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Perform a POST request with a JSON body and parse the JSON response.

        Returns:
            dict JSON on success, else None.
        """
        effective_timeout = (
            timeout_s if timeout_s is not None else self.config.timeout_s
        )
        resp = None
        try:
            resp = self.session.post(
                url,
                json=json_body,
                headers=headers,
                timeout=effective_timeout,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            self._handle_error("POST", url, json_body, e, resp)
            return None


# Convenience: module-level default client + thin wrappers.
_default_client = RestClient()


def make_api_request(
    url: str, params: Dict[str, Any], timeout: int = 10
) -> Optional[Dict[str, Any]]:
    """
    Convenience wrapper for GET requests.
    """
    return _default_client.get_json(url, params=params, timeout_s=timeout)


def make_api_post_request(
    url: str,
    json_body: Dict[str, Any],
    headers: Dict[str, str] | None = None,
    timeout: int = 10,
) -> Optional[Dict[str, Any]]:
    """
    Convenience wrapper for POST requests.
    """
    return _default_client.post_json(
        url, json_body=json_body, headers=headers, timeout_s=timeout
    )
