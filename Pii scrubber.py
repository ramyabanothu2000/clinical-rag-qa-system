"""
pii_scrubber.py
---------------
De-identifies clinical text using Microsoft Presidio and a custom spaCy NER model.
Removes / replaces PHI entities before any text reaches the embedding model or LLM.

Entities handled:
  PERSON, DATE_TIME, PHONE_NUMBER, EMAIL_ADDRESS, MEDICAL_LICENSE,
  US_SSN, US_PASSPORT, IP_ADDRESS, LOCATION, NRP, URL
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig


@dataclass
class ScrubResult:
    original_length: int
    scrubbed_text: str
    entities_found: List[dict] = field(default_factory=list)
    scrubbed_count: int = 0


class ClinicalPIIScrubber:
    """
    Wraps Presidio Analyzer + Anonymizer with healthcare-specific configuration.
    Replaces PHI with typed placeholders: <PERSON>, <DATE>, <PHONE>, etc.
    """

    ENTITIES = [
        "PERSON",
        "DATE_TIME",
        "PHONE_NUMBER",
        "EMAIL_ADDRESS",
        "MEDICAL_LICENSE",
        "US_SSN",
        "US_PASSPORT",
        "IP_ADDRESS",
        "LOCATION",
        "NRP",
        "URL",
        "CREDIT_CARD",
        "US_DRIVER_LICENSE",
    ]

    def __init__(self, language: str = "en", score_threshold: float = 0.6):
        self.language = language
        self.score_threshold = score_threshold
        self._build_engines()

    def _build_engines(self) -> None:
        nlp_config = {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_lg"}],
        }
        provider = NlpEngineProvider(nlp_configuration=nlp_config)
        nlp_engine = provider.create_engine()

        registry = RecognizerRegistry()
        registry.load_predefined_recognizers(nlp_engine=nlp_engine)

        self.analyzer = AnalyzerEngine(
            nlp_engine=nlp_engine,
            registry=registry,
            supported_languages=[self.language],
        )
        self.anonymizer = AnonymizerEngine()

    def scrub(self, text: str) -> ScrubResult:
        """
        Analyze and anonymize a single text string.

        Parameters
        ----------
        text : str
            Raw clinical text (e.g. a patient note chunk).

        Returns
        -------
        ScrubResult
            Contains scrubbed text and metadata about entities removed.
        """
        if not text or not text.strip():
            return ScrubResult(original_length=0, scrubbed_text=text)

        results = self.analyzer.analyze(
            text=text,
            entities=self.ENTITIES,
            language=self.language,
            score_threshold=self.score_threshold,
        )

        operators = {
            entity: OperatorConfig("replace", {"new_value": f"<{entity}>"})
            for entity in self.ENTITIES
        }

        anonymized = self.anonymizer.anonymize(
            text=text,
            analyzer_results=results,
            operators=operators,
        )

        entities_found = [
            {
                "entity_type": r.entity_type,
                "score": round(r.score, 3),
                "start": r.start,
                "end": r.end,
            }
            for r in results
        ]

        return ScrubResult(
            original_length=len(text),
            scrubbed_text=anonymized.text,
            entities_found=entities_found,
            scrubbed_count=len(results),
        )

    def scrub_batch(self, texts: List[str]) -> List[ScrubResult]:
        """Scrub a list of text chunks (e.g. from a chunked document)."""
        return [self.scrub(t) for t in texts]


def scrub_document(text: str, scrubber: Optional[ClinicalPIIScrubber] = None) -> str:
    """
    Convenience function. Scrubs a full document string and returns clean text.
    Creates a default scrubber if none is provided.
    """
    if scrubber is None:
        scrubber = ClinicalPIIScrubber()
    result = scrubber.scrub(text)
    return result.scrubbed_text


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sample = (
        "Patient John Doe, DOB 03/14/1978, SSN 123-45-6789, presented to "
        "Dr. Sarah Kim on 2024-01-15 with chest pain. Contact: john.doe@email.com "
        "or (555) 867-5309. Address: 123 Main St, Austin TX 78701."
    )

    scrubber = ClinicalPIIScrubber()
    result = scrubber.scrub(sample)

    print("Original :", sample)
    print("Scrubbed :", result.scrubbed_text)
    print("Entities removed:", result.scrubbed_count)
    for e in result.entities_found:
        print(f"  {e['entity_type']:25s} score={e['score']:.2f}")
