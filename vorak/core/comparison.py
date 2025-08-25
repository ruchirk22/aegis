# vorak/core/comparison.py

import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple
from .database.manager import DatabaseManager

@dataclass
class ComparisonResult:
    """Holds the comparison data for a single prompt between two sessions."""
    prompt_id: str
    category: str
    prompt_text: str
    model_name_a: str
    model_name_b: str
    classification_a: str
    classification_b: str
    score_a: float
    score_b: float
    delta: float = 0.0
    status: str = "Unchanged"

@dataclass
class ComparisonSummary:
    """Holds the high-level summary of the comparison."""
    session_a_id: str
    session_b_id: str
    total_prompts_compared: int = 0
    avg_score_a: float = 0.0
    avg_score_b: float = 0.0
    avg_score_delta: float = 0.0
    improvements: int = 0
    regressions: int = 0
    unchanged: int = 0

class ComparisonReport:
    """
    Generates a detailed comparison report between two evaluation sessions.
    """
    def __init__(self, session_a_id: str, session_b_id: str, db_manager: DatabaseManager):
        self.session_a_id = session_a_id
        self.session_b_id = session_b_id
        self.db_manager = db_manager
        self.results: List[ComparisonResult] = []
        self.summary: ComparisonSummary = ComparisonSummary(session_a_id, session_b_id)
        
        self._generate_report()

    def _fetch_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Fetches all results from the database and filters for the two sessions."""
        all_data = self.db_manager.get_all_results_as_df()
        if all_data.empty:
            raise ValueError("Database contains no results.")
            
        session_a_data = all_data[all_data['session_id'] == self.session_a_id]
        session_b_data = all_data[all_data['session_id'] == self.session_b_id]

        if session_a_data.empty:
            raise ValueError(f"Session ID '{self.session_a_id}' not found in the database.")
        if session_b_data.empty:
            raise ValueError(f"Session ID '{self.session_b_id}' not found in the database.")
            
        return session_a_data, session_b_data

    def _generate_report(self):
        """Performs the comparison and populates the results and summary."""
        df_a, df_b = self._fetch_data()

        # Merge the two dataframes on prompt_id to find common prompts
        merged_df = pd.merge(
            df_a, df_b, on="prompt_id", suffixes=('_a', '_b')
        )

        if merged_df.empty:
            raise ValueError("No common prompts found between the two sessions to compare.")

        for _, row in merged_df.iterrows():
            score_a = row['vulnerability_score_a']
            score_b = row['vulnerability_score_b']
            delta = score_b - score_a
            
            status = "Unchanged"
            if delta < 0:
                status = "Improvement"
                self.summary.improvements += 1
            elif delta > 0:
                status = "Regression"
                self.summary.regressions += 1
            else:
                self.summary.unchanged += 1

            self.results.append(ComparisonResult(
                prompt_id=row['prompt_id'],
                category=row['category_a'],
                prompt_text=row['prompt_text_a'],
                model_name_a=row['model_name_a'],
                model_name_b=row['model_name_b'],
                classification_a=row['classification_a'],
                classification_b=row['classification_b'],
                score_a=score_a,
                score_b=score_b,
                delta=delta,
                status=status
            ))

        # Calculate summary statistics
        self.summary.total_prompts_compared = len(self.results)
        self.summary.avg_score_a = merged_df['vulnerability_score_a'].mean()
        self.summary.avg_score_b = merged_df['vulnerability_score_b'].mean()
        self.summary.avg_score_delta = self.summary.avg_score_b - self.summary.avg_score_a
