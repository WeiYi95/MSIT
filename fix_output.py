# -*- coding:utf-8 -*-
# @Author: Wei Yi

input_dir = input("dir: ")
with open(input_dir+"/label_test.txt", 'r', encoding="utf-8") as file:
    lines = file.read().split('\n')
res = []
for line in lines:
    line = line.split(' ')[1:-1]
    has_bos = False
    temp = ["[CLS]"]
    for tag in line:
        if tag == "[SEP]" or tag == "[CLS]":
            temp.append("[BOS]")
            #has_bos = True
        elif tag == "[IOS]" and has_bos is False:
            temp.append("[BOS]")
            #has_bos = True
        else:
            if tag == "[BOS]":
                has_bos = True
            temp.append(tag)
    temp.append("[SEP]")
    res.append(temp)

with open(input_dir+"/pred.txt", 'w', encoding="utf-8") as file:
    for line in res:
        file.write(" ".join(line) + '\n')

