#!/usr/bin/env bash
montage cases.png edges.png deaths.png -tile 3x1 -geometry +2+2 result.png
#Montage GIF
for SR in 0.1 0.2 0.3 0.4
  do
    convert -delay 120 loop deaths_10_$SR.png  deaths_7_$SR.png deaths_5_$SR.png deaths_3_$SR.png deaths_2_$SR.png deaths_1_$SR.png deaths_$SR.gif
  done
