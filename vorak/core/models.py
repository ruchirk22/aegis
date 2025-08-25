# vorak/core/models.py

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum

class Classification(Enum):
    """
    Enum for response classification categories.
    """
    COMPLIANT = "COMPLIANT"
    NON_COMPLIANT = "NON_COMPLIANT"
    PARTIAL_COMPLIANCE = "PARTIAL_COMPLIANCE"
    ERROR = "ERROR"
    AMBIGUOUS = "AMBIGUOUS"

# --- NEW: Feature 1 - Evaluation Mode Enum ---
class EvaluationMode(str, Enum):
    """
    Enum for the different evaluation modes supported by the CLI.
    """
    STANDARD = "standard"
    ADAPTIVE = "adaptive"
    GOVERNANCE = "governance"
    ATTACK_ONLY = "attack-only"
    ANALYSIS_ONLY = "analysis-only"
    SCENARIO = "scenario"

@dataclass
class AdversarialPrompt:
    """
    A data class representing a single adversarial prompt, now with optional image data.
    """
    id: str
    category: str
    subcategory: str
    severity: str
    prompt_text: str
    expected_behavior: str
    success_indicators: List[str] = field(default_factory=list)
    failure_indicators: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    image_data: Optional[bytes] = None

    def to_dict(self) -> dict:
        """Converts the dataclass instance to a dictionary."""
        return {
            "id": self.id,
            "category": self.category,
            "subcategory": self.subcategory,
            "severity": self.severity,
            "prompt_text": self.prompt_text,
            "expected_behavior": self.expected_behavior,
            "success_indicators": self.success_indicators,
            "failure_indicators": self.failure_indicators,
            "tags": self.tags,
        }

@dataclass
class ModelResponse:
    """
    A data class to standardize the response received from an LLM.
    """
    output_text: str
    prompt_id: str
    model_name: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

@dataclass
class AnalysisResult:
    """
    A data class to hold the results of a response analysis.
    """
    classification: Classification
    explanation: str
    vulnerability_score: float
