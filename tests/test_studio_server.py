"""Regression tests for the Studio static-app/API boundary."""

from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from triune.api.server import create_app


class TestStudioServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(create_app())

    def test_studio_assets_and_api_routes_are_not_shadowed(self) -> None:
        expected = {
            "/": "text/html",
            "/app.js": "javascript",
            "/style.css": "text/css",
            "/v1/system/diagnostics": "application/json",
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


if __name__ == "__main__":
    unittest.main()
