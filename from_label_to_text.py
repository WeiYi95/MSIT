# -*- coding:utf-8 -*-
# @Author: Wei Yi


def label2text(text, label):
    res = []
    label = label.split(' ')
    idx = 0
    for i in range(len(label)):
        if label[i] != "[BOS]" and label[i] != "[IOS]":
            continue
        if idx >= len(text):
            break
        if label[i] == "[BOS]" and res != []:
            res.append(' ')
        res.append(text[idx])
        idx += 1
    return "".join(res)


text_file = input("text_file: ")
label_file = input("label file: ")
output_file = input("output file: ")
with open(text_file, 'r', encoding="utf-8") as file:
    gt = file.read().split('\n')

with open(label_file, 'r', encoding="utf-8") as file:
    pred = file.read().split('\n')

with open(output_file, 'w', encoding="utf-8") as file:
    for i in range(len(gt)):
        if gt[i] == "":
            continue
        text = gt[i].split('|')
        gt_label = text[1]
        text = text[0]
        pred_label = pred[i]
        label = label2text(text, gt_label)
        prediction = label2text(text, pred_label)
        file.write(prediction + '\n')
