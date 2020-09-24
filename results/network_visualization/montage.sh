#!/usr/bin/env bash
montage 1_pa.png 2_pa.png 3_pa.png -tile 3x1 -geometry +2+2  -title 'Preferential attachment networks' -pointsize 24 pa_nets.png
montage 1_random.png 2_random.png 3_random.png -tile 3x1 -geometry +2+2 -title 'Random networks' -pointsize 24 random_nets.png

montage pa_nets.png random_nets.png -tile 1x2 -geometry +2+2 nets.png
