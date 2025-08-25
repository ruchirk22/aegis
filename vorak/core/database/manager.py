# vorak/core/database/manager.py

import sqlite3
import pandas as pd
from typing import Dict, Any, List
from datetime import datetime

class DatabaseManager:
    """
    Manages all interactions with the local SQLite database for storing
    evaluation results.
    """
    def __init__(self, db_path: str = "vorak_results.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._create_table()

    def _create_table(self):
        """Creates the results table if it doesn't already exist."""
        cursor = self.conn.cursor()
        # --- FIX: Added the missing prompt_text column ---
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                prompt_id TEXT NOT NULL,
                category TEXT,
                prompt_text TEXT,
                model_name TEXT NOT NULL,
                model_output TEXT,
                classification TEXT NOT NULL,
                vulnerability_score REAL NOT NULL,
                explanation TEXT,
                governance_nist TEXT,
                governance_eu TEXT,
                governance_iso TEXT
            )
        """)
        self.conn.commit()

    def insert_result(self, result_data: Dict[str, Any]):
        """
        Inserts a single evaluation result into the database.
        """
        cursor = self.conn.cursor()
        # --- FIX: Added prompt_text to the INSERT statement ---
        cursor.execute("""
            INSERT INTO results (
                session_id, timestamp, prompt_id, category, prompt_text, 
                model_name, model_output, classification, vulnerability_score, 
                explanation, governance_nist, governance_eu, governance_iso
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            result_data.get("session_id", "N/A"),
            datetime.now().isoformat(),
            result_data.get("prompt_id", "N/A"),
            result_data.get("category", "N/A"),
            result_data.get("prompt_text", ""),
            result_data.get("model_name", "N/A"),
            result_data.get("model_output", ""),
            result_data.get("classification", "ERROR"),
            result_data.get("vulnerability_score", 0.0),
            result_data.get("explanation", ""),
            result_data.get("governance_nist", ""),
            result_data.get("governance_eu", ""),
            result_data.get("governance_iso", "")
        ))
        self.conn.commit()

    def get_all_results_as_df(self) -> pd.DataFrame:
        """Retrieves all results from the database as a pandas DataFrame."""
        return pd.read_sql_query("SELECT * FROM results", self.conn)

    def close(self):
        """Closes the database connection."""
        self.conn.close()
