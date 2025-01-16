from typing import Dict, List, Any
from llm_guard import scan_output, scan_prompt
from llm_guard.input_scanners import Anonymize, PromptInjection, Secrets
from llm_guard.input_scanners import Regex
from llm_guard.output_scanners import Deanonymize
from llm_guard.input_scanners.prompt_injection import MatchType as InjectionMatchType
from llm_guard.input_scanners.regex import MatchType as RegexMatchType
from llm_guard.vault import Vault

vault = Vault()


def create_input_scanners(scanner_configs: Dict[str, Dict[str, Any]]) -> List[Any]:
    scanner_map = {
        "Anonymize": lambda config: Anonymize(
            vault=vault,
            preamble=config.get("preamble", ""),
            allowed_names=config.get("allowed_names", []),
            hidden_names=config.get("hidden_names", []),
            entity_types=config.get("entity_types", None),
            use_faker=config.get("use_faker", False),
            threshold=config.get("threshold", 0),
            language=config.get("language", "en"),
        ),
        "PromptInjection": lambda config: PromptInjection(
            threshold=config.get("threshold", 0.5),
            match_type=getattr(InjectionMatchType, config.get("match_type", "FULL")),
        ),
        "Regex": lambda config: Regex(
            patterns=config.get("patterns", []),
            is_blocked=config.get("is_blocked", True),
            match_type=getattr(RegexMatchType, config.get("match_type", "SEARCH")),
            redact=config.get("redact", True),
        ),
        "Secrets": lambda config: Secrets(),
    }

    scanners = []
    if  scanner_configs is None or "input_scanners" not in scanner_configs:
        default_scanners = ["Anonymize", "Secrets","PromptInjection"]
        for scanner_name in default_scanners:
            try:
                scanner = scanner_map[scanner_name]({})
                if scanner is not None:
                    scanners.append(scanner)
            except Exception as e:
                raise ValueError(f"Failed to create default scanner {scanner_name}: {str(e)}")
        return scanners
    for scanner_name, config in scanner_configs["input_scanners"].items():
        if scanner_name in scanner_map:
            try:
                scanner = scanner_map[scanner_name](config or {})
                if scanner is not None:
                    scanners.append(scanner)
            except Exception as e:
                raise ValueError(f"Failed to create scanner {scanner_name}: {str(e)}")

    return scanners


def create_output_scanners(scanner_configs: Dict[str, Dict[str, Any]]) -> List[Any]:
    """Create output scanners based on configuration"""
    scanner_map = {
        "Deanonymize": lambda config: Deanonymize(vault=vault),
    }

    scanners = []
    if scanner_configs is None or "output_scanners" not in scanner_configs:
        return scanners
    for scanner_name, config in scanner_configs["output_scanners"].items():
        if scanner_name in scanner_map:
            try:
                scanner = scanner_map[scanner_name](config or {})
                if scanner is not None:
                    scanners.append(scanner)
            except Exception as e:
                raise ValueError(f"Failed to create scanner {scanner_name}: {str(e)}")

    return scanners


def scan_prompt_content(scanners: List[Any], prompt: str) -> str:
    """Scan prompt with provided input scanners"""
    if not scanners:
        return prompt

    sanitized_prompt, results_valid, results_score = scan_prompt(scanners, prompt)

    if "PromptInjection" in results_valid and not results_valid["PromptInjection"]:
        print(
            f"Potential prompt injection detected. Request blocked. Risk score: {results_score.get('PromptInjection', 0)}"
        )
        return "Give reply in single sentence that is 'Prompt Injection is detected'"

    return sanitized_prompt


def scan_output_content(scanners: List[Any], prompt: str, response: str) -> str:
    """Scan output with provided output scanners"""
    if not scanners:
        return response

    sanitized_response, _, _ = scan_output(scanners, prompt, response)

    return sanitized_response
