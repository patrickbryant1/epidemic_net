#!/usr/bin/env bash

for m in 1 2 3 4 5
  do
    montage cases_$m'_0-19.png' cases_$m'_20-49.png' cases_$m'_50-69.png' cases_$m'_70+.png' -tile 1x4 -geometry +2+2 cases_$m.png
  done

montage cases_1.png cases_2.png cases_3.png cases_4.png cases_5.png -tile 5x1 -geometry +2+2 -title 'New York' -pointsize 24 all_cases.png

montage cases_1_total.png cases_2_total.png cases_3_total.png cases_4_total.png cases_5_total.png ../deaths/markers.png -tile 3x2 -geometry +2+2 -title 'New York' -pointsize 24 total_cases.png
