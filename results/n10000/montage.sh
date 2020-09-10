#!/usr/bin/env bash
#montage cases.png edges.png deaths.png -tile 3x1 -geometry +2+2 result.png
#Montage GIF
for SR in '2_2_2_2_2_2' '2_2_3_3_3_3' '2_2_4_4_4_4' '2_2_5_5_5_5'
  do
    convert -delay 120 loop deaths_5_$SR.png deaths_3_$SR.png deaths_2_$SR.png deaths_1_$SR.png deaths_$SR.gif
  done
