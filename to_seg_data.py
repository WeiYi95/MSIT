# -*- coding:utf-8 -*-
# -*- coding:utf-8 -*-
# @Author: Wei Yi

import random
MAX_LEN = 0
def to_label(line):
    words = line.split(' ')
    label = []
    for word in words:
        label.append("[BOS]")
        rem = len(word) - 1
        for _ in range(rem):
            label.append("[IOS]")
    return "".join(words), " ".join(label)

seg_lines = []
input_file = input("Input filename: ")
output_file = input("Output filename: ")
with open(input_file, 'r', encoding="utf-8") as file:
    lines = file.read().split('\n')
    for line in lines:
        line, label = to_label(line)
        MAX_LEN = max(MAX_LEN, len(line))
        if len(line) > 22:
            continue
        seg_lines.append(line + '|' + label)
print(MAX_LEN)
with open(output_file, 'w', encoding="utf-8") as file:
    for seg in seg_lines:
        file.write(seg + '\n')
