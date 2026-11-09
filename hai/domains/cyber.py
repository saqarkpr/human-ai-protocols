"""Cybersecurity alert triage.

Real NSL-KDD network connection records (Tavallaee et al., 2009), see
data/cyber/PROVENANCE.md. KDDTrain+ is split into training and optimisation
parts. KDDTest+ is the held-out evaluation set. It contains attack types that
never occur in training, so it is also a test under distribution shift.

Prevalence. Attacks make up about half of NSL-KDD, which is unrealistic for
alert triage. Attacks are subsampled (fixed seed) to 20% of each split.

Information split. The AI is a flow-level intrusion detector. It sees the
basic connection features and the traffic statistics, but NOT the content
features that require inspecting the session on the host (failed logins,
root shells, file creations, "hot" indicators and so on). The simulated
analyst judges from those host and session cues, plus whether the session
was logged in and whether the connection completed normally.
"""
import gzip
from pathlib import Path

import numpy as np
import pandas as pd

from ..protocols import Costs
from .base import Domain

DATA = Path(__file__).resolve().parents[2] / "data" / "cyber"
COLUMNS = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
           "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised",
           "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells",
           "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
           "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
           "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
           "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
           "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
           "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
           "attack", "difficulty"]
CONTENT = ["hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell",
           "su_attempted", "num_root", "num_file_creations", "num_shells", "num_access_files",
           "num_outbound_cmds", "is_host_login", "is_guest_login"]
CATEGORICAL = ["protocol_type", "service", "flag"]
PREVALENCE = 0.20


def _read(name):
    with gzip.open(DATA / name, "rt") as f:
        df = pd.read_csv(f, header=None, names=COLUMNS)
    df["is_attack"] = (df["attack"] != "normal").astype(int)
    return df


def _set_prevalence(df, rng, n_max=None):
    pos, neg = df[df.is_attack == 1], df[df.is_attack == 0]
    if n_max is not None and len(neg) > n_max * (1 - PREVALENCE):
        neg = neg.sample(int(n_max * (1 - PREVALENCE)), random_state=int(rng.integers(1 << 30)))
    n_pos = int(round(len(neg) * PREVALENCE / (1 - PREVALENCE)))
    pos = pos.sample(min(n_pos, len(pos)), random_state=int(rng.integers(1 << 30)))
    return pd.concat([pos, neg]).sample(frac=1, random_state=int(rng.integers(1 << 30))).reset_index(drop=True)


class CyberDomain(Domain):
    def splits(self):
        if not self._cache:
            rng = np.random.default_rng(self.seed)
            tr = _read("KDDTrain+.txt.gz")
            te = _read("KDDTest+.txt.gz")
            idx = rng.permutation(len(tr))
            n_opt = int(0.3 * len(tr))
            opt, train = tr.iloc[idx[:n_opt]], tr.iloc[idx[n_opt:]]
            self._cache.update(train=_set_prevalence(train, rng, n_max=20000),
                               opt=_set_prevalence(opt, rng, n_max=4000),
                               eval=_set_prevalence(te, rng))
            self._levels = {c: sorted(self._cache["train"][c].unique()) for c in CATEGORICAL}
        return self._cache

    def ai_features(self, df, private=False):
        self.splits()
        X = df[[c for c in COLUMNS[:41] if private or c not in CONTENT]].copy()
        for c in CATEGORICAL:
            for lvl in self._levels[c]:
                X[f"{c}={lvl}"] = (X[c] == lvl).astype(int)
            X = X.drop(columns=c)
        return X

    def cues(self, df):
        X = df[CONTENT].copy()
        X["completed_normally"] = (df["flag"] == "SF").astype(int)
        return X.drop(columns=["num_outbound_cmds", "is_host_login"])

    def label(self, df):
        return df["is_attack"].values

    def severity(self, df):
        """A rule-based legacy priority score (the kind of signature count a
        SIEM would show), observable before any human or AI review."""
        return ((df["serror_rate"] > 0.5).astype(int) + (df["rerror_rate"] > 0.5).astype(int)
                + (df["wrong_fragment"] > 0).astype(int) + (df["num_failed_logins"] > 0).astype(int)
                + (df["hot"] > 0).astype(int) + (df["root_shell"] > 0).astype(int)
                + (df["count"] > 100).astype(int)).values.astype(float)


CYBER = CyberDomain(name="cyber", costs=Costs(fn=10.0, fp=1.0), case="connection",
                    positive="intrusion",
                    human_private="host and session content (logins, shells, file activity)")
