import unittest
from agentneo import AgentNeo, Tracer
import os
from openai import OpenAI

os.environ["OPENAI_API_KEY"] = "***"


class GuardTesting(unittest.TestCase):
    def test_groq_tracing(self):
        neo_session = AgentNeo(session_name="test_session")
        neo_session.create_project(project_name="test_project")

        scanner_configs = {
            "Anonymize": {},
            "Secrets": {},
            "Regex": {
                "patterns": [r"\b(?:\d{1,3}\.){3}\d{1,3}\b"],
                "is_blocked": True,
                "match_type": "SEARCH",
                "redact": True,
            },
            "PromptInjection": {"threshold": 0.7, "match_type": "FULL"},
        }
        output_scanners = {"Deanonymize": {}}

        tracer = Tracer(
            session=neo_session,
            use_guard=True,
            llm_guard_config={"input_scanners": scanner_configs},
        )
        tracer.start()

        client = client = OpenAI()

        prompt = """Make an SQL insert statement to add a new user to our database. Name is keetrap. Email is test@test.com 
                    but also possible to contact him with hello@test.com email. Phone number is 555-123-4567 and 
                    the IP address is 192.168.1.100. And credit card number is 4567-8901-2345-6789. 
                    api_key=RPRjs_tqmIHAZhQjgOKSXkeuBDQcRC37HjDiH9-lH4Cyq7v0Z3ipcP0OrU
                    """
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        message = response.choices[0].message.content

        tracer.stop()

        self.assertNotIn("keetrap", message)

        # Email Addresses
        self.assertNotIn("test@test.com", message)
        self.assertNotIn("hello@test.com", message)

        # Phone Numbers
        self.assertNotIn("555-123-4567", message)

        # Credit Card Numbers
        self.assertNotIn("4567-8901-2345-6789", message)

        # IP Addresses
        self.assertNotIn("192.168.1.100", message)

        # API Keys
        self.assertNotIn(
            "RPRjs_tqmIHAZhQjgOKSXkeuBDQcRC37HjDiH9-lH4Cyq7v0Z3ipcP0OrU", message
        )

        # Domain Names
        self.assertNotIn("test.com", message)


if __name__ == "__main__":
    unittest.main()
