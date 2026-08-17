import os
import unittest

from efficient_rlm.config import RLMConfig
from efficient_rlm.llm.http import HTTPGenerationClient, build_llm_client
from efficient_rlm.llm.mock import MockLLMClient


class ProviderConfigTests(unittest.TestCase):
    def test_mock_provider_imports_without_credentials(self):
        client = build_llm_client(RLMConfig(provider="mock"))
        self.assertIsInstance(client, MockLLMClient)

    def test_ollama_provider_does_not_require_api_key(self):
        client = build_llm_client(RLMConfig(provider="ollama", model="example"))
        self.assertIsInstance(client, HTTPGenerationClient)

    def test_openai_compatible_requires_endpoint(self):
        with self.assertRaisesRegex(ValueError, "endpoint"):
            build_llm_client(RLMConfig(provider="openai_compatible"))

    def test_openai_compatible_requires_api_key_env(self):
        old_value = os.environ.pop("RLM_API_KEY", None)
        try:
            with self.assertRaisesRegex(ValueError, "RLM_API_KEY"):
                build_llm_client(
                    RLMConfig(
                        provider="openai_compatible",
                        endpoint="https://api.example.test/v1/chat/completions",
                    )
                )
        finally:
            if old_value is not None:
                os.environ["RLM_API_KEY"] = old_value


if __name__ == "__main__":
    unittest.main()

