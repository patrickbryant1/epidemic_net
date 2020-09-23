#!/usr/bin/env

#Total
montage deaths_1_2_2_2_2_total.png deaths_1_3_3_3_3_total.png deaths_1_4_4_4_4_total.png markers.png -tile 4x1 -geometry +2+2 -title 'Spain' -pointsize 24 total_deaths.png
montage deaths_1_total_100.png deaths_2_total_100.png deaths_3_total_100.png deaths_4_total_100.png deaths_5_total_100.png markers_100.png -tile 3x2 -geometry +2+2 -title 'Spain' -pointsize 24 total_deaths_100.png
