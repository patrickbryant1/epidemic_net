#!/usr/bin/env bash

for m in 1 2 3 4 5
  do
    montage deaths_$m'_0-19.png' deaths_$m'_20-49.png' deaths_$m'_50-69.png' deaths_$m'_70+.png' -tile 1x4 -geometry +2+2 deaths_$m.png
  done

montage deaths_1.png deaths_2.png deaths_3.png deaths_4.png deaths_5.png -tile 5x1 -geometry +2+2 -title 'New York City' -pointsize 24 all_deaths.png

montage deaths_1_total.png deaths_2_total.png deaths_3_total.png deaths_4_total.png deaths_5_total.png markers.png -tile 3x2 -geometry +2+2 -title 'New York City' -pointsize 24 total_deaths.png
