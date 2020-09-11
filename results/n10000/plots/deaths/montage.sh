#!/usr/bin/env bash

for m in 1 2 3 4 5
  do
    montage deaths_$m'_0-49.png' deaths_$m'_50-59.png' deaths_$m'_60-69.png' deaths_$m'_70-79.png' deaths_$m'_80-89.png' deaths_$m'_90+.png' -tile 1x7 -geometry +2+2 deaths_$m.png
  done

montage deaths_1.png deaths_2.png deaths_3.png deaths_4.png deaths_5.png -tile 5x1 -geometry +2+2 all_deaths.png

montage deaths_1_total.png deaths_2_total.png deaths_3_total.png deaths_4_total.png deaths_5_total.png markers.png -tile 3x2 -geometry +2+2 total_deaths.png
