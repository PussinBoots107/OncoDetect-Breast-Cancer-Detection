import unittest
from app import app, FEATURE_NAMES, DEMO_CASES, MODELS

class OncoDetectSmokeTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_health(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["features"], 30)

    def test_home_renders(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"OncoDetect", response.data)

    def test_all_models_predict_demo(self):
        for model_name in MODELS:
            payload = {"model": model_name, "features": DEMO_CASES["benign_demo"]}
            response = self.client.post("/api/predict", json=payload)
            self.assertEqual(response.status_code, 200, (model_name, response.get_json()))
            self.assertIn(response.get_json()["prediction"], [0, 1])

    def test_missing_features_rejected(self):
        response = self.client.post("/api/predict", json={"model": "Support Vector Machine", "features": {}})
        self.assertEqual(response.status_code, 400)

    def test_unknown_model_rejected(self):
        response = self.client.post("/api/predict", json={"model": "Mystery", "features": DEMO_CASES["benign_demo"]})
        self.assertEqual(response.status_code, 400)

if __name__ == "__main__":
    unittest.main()
