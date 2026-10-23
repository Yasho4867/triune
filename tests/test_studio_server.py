"""Regression tests for the Studio static-app/API boundary."""

from __future__ import annotations

import unittest
import inspect
from unittest.mock import patch

from fastapi.testclient import TestClient

from triune.api.server import create_app


class TestStudioServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        client_kwargs = {}
        if "client" in inspect.signature(TestClient).parameters:
            client_kwargs["client"] = ("127.0.0.1", 43210)
        cls.client = TestClient(create_app(), **client_kwargs)

    def test_studio_assets_and_api_routes_are_not_shadowed(self) -> None:
        expected = {
            "/": "text/html",
            "/app.js": "javascript",
            "/style.css": "text/css",
            "/v1/system/diagnostics": "application/json",
            "/healthz": "application/json",
            "/v1/nodes/catalog": "application/json",
        }
        for path, content_type in expected.items():
            with self.subTest(path=path):
                response = self.client.get(path)
                self.assertEqual(response.status_code, 200)
                self.assertIn(content_type, response.headers["content-type"])

    def test_node_catalog_is_populated_for_the_visual_canvas(self) -> None:
        response = self.client.get("/v1/nodes/catalog")
        catalog = response.json()
        self.assertGreaterEqual(len(catalog), 30)
        self.assertTrue(any(node["name"] == "TriuneTransformer" for node in catalog))

    def test_spa_fallback_does_not_serve_files_outside_studio_assets(self) -> None:
        response = self.client.get("/%2e%2e/pyproject.toml")
        self.assertEqual(response.status_code, 200)
        self.assertNotIn('name = "triune"', response.text)

    def test_remote_api_access_requires_a_valid_bearer_token(self) -> None:
        if "client" not in inspect.signature(TestClient).parameters:
            self.skipTest("Installed Starlette TestClient cannot override the client IP")
        remote_client = TestClient(create_app(), client=("203.0.113.10", 43210))
        with patch.dict("os.environ", {"TRIUNE_API_TOKEN": ""}):
            unconfigured = remote_client.get("/v1/nodes/catalog")
        self.assertEqual(unconfigured.status_code, 503)

        api_token = "t" * 40
        with patch.dict("os.environ", {"TRIUNE_API_TOKEN": api_token}):
            unauthorized = remote_client.get("/v1/nodes/catalog")
            allowed = remote_client.get("/v1/nodes/catalog", headers={"Authorization": f"Bearer {api_token}"})
        self.assertEqual(unauthorized.status_code, 401)
        self.assertEqual(allowed.status_code, 200)


if __name__ == "__main__":
    unittest.main()
