#!/usr/bin/env bash
montage cases.png cumulative_cases.png deaths.png active_spreaders.png -tile 2x2 -geometry +2+2 result.png
