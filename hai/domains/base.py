from dataclasses import dataclass, field

import numpy as np

from ..protocols import Costs


@dataclass
class Domain:
    """Interface every domain implements.

    splits()                     -> {"train", "opt", "eval"} DataFrames
    ai_features(df, private)     -> what the AI sees (private=True also gives it
                                    the human's private information, for the ablation)
    cues(df)                     -> bedside / analyst cues of the human agents
    label(df)                    -> 0/1 outcome
    severity(df)                 -> an observable triage-severity score that a
                                    status-quo routing policy would use (causal analysis)
    """
    name: str
    costs: Costs
    case: str            # "patient" or "connection"
    positive: str        # "death" or "intrusion"
    human_private: str   # description of what only the human sees
    seed: int = 0
    _cache: dict = field(default_factory=dict, repr=False)

    def severity_bins(self, df, cuts=None):
        s = np.asarray(self.severity(df), float)
        s = np.where(np.isnan(s), np.nanmedian(s), s)
        if cuts is None:
            cuts = np.unique(np.nanquantile(s, [1 / 3, 2 / 3]))
        b = np.digitize(s, cuts, right=True)
        return b, cuts
