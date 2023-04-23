#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test utils"""

from model.utils import clean_maxspeed
import pandas as pd


def test_clean_maxspeed():
    """Tests if maxspeed is cleaned properly"""

    test_maxspeed = pd.DataFrame({"maxspeed": ["30", 34.0, 10, "none", "5 mph"]})
    expected_result = [30.0, 34.0, 10.0, None, None]
    result = clean_maxspeed(test_maxspeed["maxspeed"])
    assert [r == e for r, e in zip(result, expected_result)]
