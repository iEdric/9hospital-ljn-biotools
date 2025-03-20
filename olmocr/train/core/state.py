import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class BeakerState:
    job_id: Optional[str] = None
    job_kind: Optional[str] = None
    task_id: Optional[str] = None
    experiment_id: Optional[str] = None
    replica_rank: Optional[str] = None
    leader_replica_hostname: Optional[str] = None
    leader_replica_node_id: Optional[str] = None
    user_id: Optional[str] = None

    def __post_init__(self):
        for key, value in os.environ.items():
            if not key.startswith("BEAKER_"):
                continue
            setattr(self, key.lstrip("BEAKER_").lower(), value)

    @property
    def url(self) -> Optional[str]:
        if self.job_id:
            return f"https://beaker.org/jobs/{self.job_id}"
        return None
