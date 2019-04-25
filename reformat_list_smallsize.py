from __future__ import print_function, division
import sys
import os
import re


out_lines = []
'''
with open(sys.argv[1], 'r') as the_file:
    lines = list(the_file)
    for line in lines:
        new_line = re.sub("\.jp\n", "\.jpg\n", line)
        out_lines.append(new_line)
'''

with open(sys.argv[1], 'r') as the_file:
    lines = list(the_file)
    for line in lines:
        new_line = re.sub("g\/video", "g\n/video", line)
        out_lines.append(new_line)

with open(sys.argv[1], 'w') as the_file:
    for line in out_lines:
        the_file.write(line + '\n')
