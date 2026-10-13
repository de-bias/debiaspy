"""Multi-method contracts based on debiasR's validation helper fixtures."""

from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def inputs():
    raw = pd.DataFrame(
        {"origin": list("AABB"), "destination": list("XYXY"), "flow": [100, 50, 25, 75]}
    )
    benchmark = raw.assign(flow=[90, 65, 35, 70])
    methods = {
        "method_a": raw.assign(flow_adj=[92, 62, 33, 72]),
        "method_b": raw.assign(flow_adj=[96, 58, 30, 78]),
    }
    return raw, benchmark, methods
