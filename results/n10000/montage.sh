#!/usr/bin/env bash
montage cases.png edges.png deaths.png -tile 3x1 -geometry +2+2 result.png
#Montage GIF
convert -delay 120 loop deaths_15.png deaths_10.png  deaths_7.png deaths_5.png deaths_3.png deaths_2.png deaths_1.png deaths_0.png deaths.gif
