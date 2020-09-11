#!/usr/bin/env bash

for m in 1 2 3 4 5
  do
    montage cases_$m'_0-49.png' cases_$m'_50-59.png' cases_$m'_60-69.png' cases_$m'_70-79.png' cases_$m'_80-89.png' cases_$m'_90+.png' -tile 1x7 -geometry +2+2 cases_$m.png
  done

montage cases_1.png cases_2.png cases_3.png cases_4.png cases_5.png -tile 5x1 -geometry +2+2 all_cases.png

montage cases_1_total.png cases_2_total.png cases_3_total.png cases_4_total.png cases_5_total.png markers.png -tile 3x2 -geometry +2+2 total_cases.png
