# vorak/core/governance.py

from typing import List, Dict, Optional
from .models import AdversarialPrompt, GovernanceResult

class GovernanceMapper:
    """
    Maps vulnerability categories to major AI governance and compliance frameworks.
    """
    MAPPINGS: Dict[str, Dict[str, List[str]]] = {
        "Bias_Harmful_Content": {
            "NIST AI RMF": ["Validity and Reliability", "Safety", "Accountability and Transparency"],
            "EU AI Act": ["Article 10 (Data and data governance)", "Article 13 (Transparency)"],
            "ISO/IEC 23894": ["Risk Management", "Transparency and Explainability"],
            "MITRE ATLAS": ["AML.T0004: Skew Training Data", "AML.T0005: Introduce Biased Data"]
        },
        "Code_Generation_Exploits": {
            "NIST AI RMF": ["Security", "Safety"],
            "EU AI Act": ["Article 15 (Accuracy, robustness and cybersecurity)"],
            "ISO/IEC 23894": ["Security", "Robustness"],
            "MITRE ATLAS": ["AML.T0001: Prompt Injection"]
        },
        "Data_Exfiltration": {
            "NIST AI RMF": ["Security", "Privacy"],
            "EU AI Act": ["Article 10 (Data and data governance)", "Article 15 (Cybersecurity)"],
            "ISO/IEC 23894": ["Security", "Data Quality"],
            "MITRE ATLAS": ["AML.T0003: Model Output Exfiltration"]
        },
        "Direct_Prompt_Injection": {
            "NIST AI RMF": ["Security", "Accountability and Transparency"],
            "EU AI Act": ["Article 15 (Robustness and cybersecurity)"],
            "ISO/IEC 23894": ["Security"],
            "MITRE ATLAS": ["AML.T0001: Prompt Injection"]
        },
        "Indirect_Prompt_Injection": {
            "NIST AI RMF": ["Security", "Safety"],
            "EU AI Act": ["Article 15 (Robustness and cybersecurity)"],
            "ISO/IEC 23894": ["Security"],
            "MITRE ATLAS": ["AML.T0001: Prompt Injection"]
        },
        "Jailbreaking_Role-Playing": {
            "NIST AI RMF": ["Safety", "Accountability and Transparency"],
            "EU AI Act": ["Article 9 (Risk management system)"],
            "ISO/IEC 23894": ["Risk Management"],
            "MITRE ATLAS": ["AML.T0001: Prompt Injection"]
        },
        "Misinformation_Deception": {
            "NIST AI RMF": ["Validity and Reliability", "Accountability and Transparency"],
            "EU AI Act": ["Article 52 (Transparency obligations for certain AI systems)"],
            "ISO/IEC 23894": ["Transparency and Explainability"],
            "MITRE ATLAS": ["AML.T0006: Generate Misinformation"]
        },
        "Privacy_Confidentiality": {
            "NIST AI RMF": ["Privacy", "Security"],
            "EU AI Act": ["Article 10 (Data and data governance)"],
            "ISO/IEC 23894": ["Data Quality", "Security"],
            "MITRE ATLAS": ["AML.T0003: Model Output Exfiltration"]
        }
    }

    def get_governance_risks(self, prompt: AdversarialPrompt) -> Optional[GovernanceResult]:
        """
        Gets the governance and compliance mappings for a given prompt's category.
        """
        category_mappings = self.MAPPINGS.get(prompt.category)
        if not category_mappings:
            return None
        
        return GovernanceResult(
            nist_ai_rmf=category_mappings.get("NIST AI RMF", []),
            eu_ai_act=category_mappings.get("EU AI Act", []),
            iso_iec_23894=category_mappings.get("ISO/IEC 23894", []),
            mitre_atlas=category_mappings.get("MITRE ATLAS", [])
        )
