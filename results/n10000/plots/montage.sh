#!/usr/bin/env bash
#montage cases.png edges.png deaths.png -tile 3x1 -geometry +2+2 result.png
#Montage GIF
for m in 5 3 2 1
  do
    convert -delay 120 loop deaths_$m'_1_1_1_1_1_1'.png deaths_$m'_2_2_2_2_2_2'.png deaths_$m'_2_2_3_3_3_3'.png deaths_$m'_2_2_4_4_4_4'.png deaths_$m'_2_2_5_5_5_5'.png deaths_$m.gif
      convert -delay 120 loop weekly_cases_$m'_1_1_1_1_1_1'.png weekly_cases_$m'_2_2_2_2_2_2'.png weekly_cases_$m'_2_2_3_3_3_3'.png weekly_cases_$m'_2_2_4_4_4_4'.png weekly_cases_$m'_2_2_5_5_5_5'.png weekly_cases_$m.gif
  done
