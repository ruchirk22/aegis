# vorak/core/evaluators/base.py

# FIX: Import ABC to mark this as an abstract class
from abc import ABC, abstractmethod
from ..models import ModelResponse, AnalysisResult
from ..plugins import vorakPlugin

# FIX: Inherit from ABC
class Evaluator(vorakPlugin, ABC):
    """
    Abstract Base Class for all evaluator modules.
    
    Each evaluator is responsible for a specific type of analysis
    on an LLM's response.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """The unique name of the evaluator."""
        pass

    @abstractmethod
    def analyze(self, response: ModelResponse) -> AnalysisResult:
        """
        Performs the analysis on the model's response.
        """
        pass
