# vorak/core/models.py

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum

class Classification(Enum):
    """Enum for response classification categories."""
    COMPLIANT = "COMPLIANT"
    NON_COMPLIANT = "NON_COMPLIANT"
    PARTIAL_COMPLIANCE = "PARTIAL_COMPLIANCE"
    ERROR = "ERROR"
    AMBIGUOUS = "AMBIGUOUS"

class EvaluationMode(str, Enum):
    """Enum for the different evaluation modes supported by the CLI."""
    STANDARD = "standard"
    ADAPTIVE = "adaptive"
    GOVERNANCE = "governance"
    ATTACK_ONLY = "attack-only"
    ANALYSIS_ONLY = "analysis-only"
    SCENARIO = "scenario"

# --- NEW: Feature 5 - Dataclass for Governance Results ---
@dataclass
class GovernanceResult:
    """A data class to hold governance and compliance mapping results."""
    nist_ai_rmf: List[str] = field(default_factory=list)
    eu_ai_act: List[str] = field(default_factory=list)
    iso_iec_23894: List[str] = field(default_factory=list)

@dataclass
class AdversarialPrompt:
    """A data class representing a single adversarial prompt."""
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
            "id": self.id, "category": self.category, "subcategory": self.subcategory,
            "severity": self.severity, "prompt_text": self.prompt_text,
            "expected_behavior": self.expected_behavior, "success_indicators": self.success_indicators,
            "failure_indicators": self.failure_indicators, "tags": self.tags,
        }

@dataclass
class ModelResponse:
    """A data class to standardize the response received from an LLM."""
    output_text: str
    prompt_id: str
    model_name: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

# --- MODIFIED: Feature 5 - Add governance results to the main analysis ---
@dataclass
class AnalysisResult:
    """A data class to hold the results of a response analysis."""
    classification: Classification
    explanation: str
    vulnerability_score: float
    governance: Optional[GovernanceResult] = None
