import numpy as np
import pandas as pd

from hdblens.train import BaselineModel


def test_baseline_lookup_and_fallback():
    train = pd.DataFrame(
        {
            "town": ["BEDOK", "BEDOK", "YISHUN"],
            "flat_type": ["4 ROOM", "4 ROOM", "3 ROOM"],
            "resale_price": [500000, 520000, 380000],
        }
    )
    model = BaselineModel().fit(train)

    seen = pd.DataFrame({"town": ["BEDOK"], "flat_type": ["4 ROOM"]})
    assert model.predict(seen)[0] == 510000  # median of the two

    unseen = pd.DataFrame({"town": ["PUNGGOL"], "flat_type": ["5 ROOM"]})
    assert model.predict(unseen)[0] == 500000  # global median fallback


def test_baseline_never_returns_nan():
    train = pd.DataFrame(
        {"town": ["BEDOK"], "flat_type": ["4 ROOM"], "resale_price": [500000]}
    )
    model = BaselineModel().fit(train)
    unseen = pd.DataFrame({"town": ["MARS"], "flat_type": ["10 ROOM"]})
    assert not np.isnan(model.predict(unseen)).any()
