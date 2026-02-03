from typing import Dict, Optional


class PolicyService:
    """Mock policy data service (later replaceable with real REST API)"""

    def __init__(self):
        self._policies = {
            "POLICY123": {
                "holder": "John Doe",
                "type": "Auto",
                "coverage": "Comprehensive",
                "deductible": 500,
                "status": "Active"
            },
            "POLICY456": {
                "holder": "Jane Smith",
                "type": "Home",
                "coverage": "All-risk",
                "deductible": 1000,
                "status": "Active"
            },
        }

    def get_policy(self, policy_id: str) -> Optional[Dict]:
        return self._policies.get(policy_id.upper())