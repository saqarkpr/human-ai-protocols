"""Application domains. Each domain provides real data and decides which
information the AI sees and which the human sees."""
from .icu import ICU
from .cyber import CYBER

DOMAINS = {"icu": ICU, "cyber": CYBER}
