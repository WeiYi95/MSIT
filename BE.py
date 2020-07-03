# -*- coding:utf-8 -*-
# @Author: Wei Yi

import math
import os
import utils
from collections import defaultdict


front_ngram = defaultdict(int)
back_ngram = defaultdict(int)
front_ngram_sp = dict()
back_ngram_sp = dict()
front_ngram_cnt = defaultdict(int)
front_ngram_tot = defaultdict(float)
front_ngram_avg = defaultdict(float)
back_ngram_cnt = defaultdict(int)
back_ngram_tot = defaultdict(float)
back_ngram_avg = defaultdict(float)
front_ngram_BE = defaultdict(float)
back_ngram_BE = defaultdict(float)
front_ngram_VBE = defaultdict(float)
back_ngram_VBE = defaultdict(float)
bt = set()

DEBUG = False


def write_txt(fn, text_list):
    sent2sent = dict()
    for s in text_list:
        unseg = "".join(s.split(" "))
        sent2sent[unseg] = s
    utils.save_as_pkl(fn, sent2sent)


def init(corpus=None, fn=None, redo=False, use_jieba=False):
    global front_ngram_sp, front_ngram, front_ngram_avg, front_ngram_BE, front_ngram_tot, front_ngram_cnt, front_ngram_VBE
    global back_ngram_cnt, back_ngram_tot, back_ngram_avg, back_ngram_sp, back_ngram_BE, back_ngram, back_ngram_VBE
    if os.path.exists("seg_BE.pkl") and os.path.exists("seg_nVBE.pkl"):
        if not redo: return
    if os.path.exists("stat.pkl") and not redo:
        print("Loading stat ...")
        front_ngram_avg, back_ngram_avg, front_ngram_BE, back_ngram_BE, front_ngram_VBE, back_ngram_VBE \
            = utils.read_from_pkl("stat.pkl")
    else:
        print("Getting stat ...")
        run_stat(corpus)
        utils.save_as_pkl("stat.pkl", [front_ngram_avg, back_ngram_avg, front_ngram_BE, back_ngram_BE, front_ngram_VBE,
                                       back_ngram_VBE])
    seg_BE(fn)
    seg_nVBE(fn)

    if use_jieba:
        import jieba
        with open(fn, 'r', encoding="utf-8") as file:
            lines = [line for line in file.read().split("\n") if len(line) > 0]

        s2s = dict()
        for line in lines:
            s2s[line] = " ".join(jieba.cut(line))
        output_pkl = open("seg_jieba.pkl", 'wb')
        pickle.dump(s2s, output_pkl)
        output_pkl.close()


def run_stat(fn):
    global front_ngram_sp, front_ngram, front_ngram_avg, front_ngram_BE, front_ngram_tot, front_ngram_cnt
    global back_ngram_cnt, back_ngram_tot, back_ngram_avg, back_ngram_sp, back_ngram_BE, back_ngram
    with open(fn, 'r', encoding="utf-8") as file:
        lines = [line for line in file.read().split("\n") if len(line) > 0]
    for line in lines:
        for st in range(0, len(line)-1):
            for i in range(1, len(line)):
                if i < st: continue
                cur, nex = line[st:i], line[i]
                if cur not in front_ngram_sp.keys():
                    front_ngram_sp[cur] = defaultdict(int)
                front_ngram_sp[cur][nex] += 1
                front_ngram[cur] += 1

    for line in lines:
        for offset in range(0, len(line) - 1):
            for i in range(1, len(line)):
                if i < offset: continue
                cur, nex = line[len(line)-i:len(line)-offset], line[len(line)-i-1]
                if cur not in back_ngram_sp.keys():
                    back_ngram_sp[cur] = defaultdict(int)
                back_ngram_sp[cur][nex] += 1
                back_ngram[cur] += 1

    # if DEBUG: print(back_ngram); print("="*20); print(back_ngram_sp); print("="*20); \
    #                              print(front_ngram); print("="*20); print(front_ngram_sp); print("="*20);

    for ng, kvd in front_ngram_sp.items():
        tot = front_ngram[ng]
        be = 0.0
        for k, v in kvd.items():
            ent = -1 * (v / tot) * math.log(v / tot, 2)
            be += ent
        front_ngram_BE[ng] = be

    for ng, vbe in front_ngram_BE.items():
        if len(ng) != 1:
            vbe -= front_ngram_BE[ng[:-1]]
        front_ngram_VBE[ng] = vbe
        front_ngram_tot[len(ng)] += vbe
        front_ngram_cnt[len(ng)] += 1

    for ng, kvd in back_ngram_sp.items():
        tot = back_ngram[ng]
        be = 0.0
        for k, v in kvd.items():
            ent = -1 * (v / tot) * math.log(v / tot, 2)
            be += ent
        back_ngram_BE[ng] = be

    for ng, vbe in back_ngram_BE.items():
        if len(ng) != 1:
            vbe -= back_ngram_BE[ng[1:]]
        back_ngram_VBE[ng] = vbe
        back_ngram_tot[len(ng)] += vbe
        back_ngram_cnt[len(ng)] += 1

    # if DEBUG: print(back_ngram_tot); print("="*20); print(front_ngram_tot); print("="*20);
    # if DEBUG: print(back_ngram_cnt); print("=" * 20); print(front_ngram_cnt); print("=" * 20);

    for k in back_ngram_tot.keys():
        back_ngram_avg[k] = back_ngram_tot[k] / back_ngram_cnt[k]
    for k in front_ngram_tot.keys():
        front_ngram_avg[k] = front_ngram_tot[k] / front_ngram_cnt[k]


def seg_BE(fn):
    global front_ngram_BE, back_ngram_BE
    with open(fn, 'r', encoding="utf-8") as file:
        lines = [line for line in file.read().split("\n") if len(line) > 0]
    save_name = "seg_BE.pkl"
    seg_lines = []
    for line in lines:
        seg_p = set()
        last_be = 1 << 31
        st_idx = 0

        if DEBUG: print(line)

        for i in range(1, len(line)+1):
            cur = line[st_idx:i]
            be = front_ngram_BE[cur] if cur in front_ngram_BE.keys() else (1 << 31) +1

            if DEBUG:
                print(cur, last_be, be, cur in front_ngram_BE.keys())
                _ = input("bp:")

            if be > last_be:
                seg_p.add(i-1)
                st_idx = i

                if DEBUG: print(i-1)

                last_be = 1<<31
            else:
                last_be = be

        last_be = 1 << 31
        fi_idx = len(line)
        for i in range(1, len(line)+1):
            cur = line[len(line)-i:fi_idx]
            be = back_ngram_BE[cur] if cur in back_ngram_BE.keys() else (1 << 31) +1

            if DEBUG:
                print(cur, last_be, be, cur in front_ngram_BE.keys())
                _ = input("bp:")

            if be > last_be:
                seg_p.add(len(line)-i+1)

                if DEBUG: print(len(line)-i+1)

                fi_idx = len(line)-i+1
                last_be = 1 << 31
            else:
                last_be = be

        seg_l = list()
        for i in range (0, len(line)):
            if i in seg_p and i != 0:
                seg_l.append(" ")
            seg_l.append(line[i])

        if DEBUG: print(seg_p, seg_l)

        seg_lines.append("".join(seg_l))

    write_txt(save_name, seg_lines)


def seg_nVBE(fn):
    global bt
    with open(fn, 'r', encoding="utf-8") as file:
        lines = [line for line in file.read().split("\n") if len(line) > 0]
    save_name = "seg_nVBE.pkl"
    seg_lines = []
    for line in lines:
        dp_mat = dp_cut(line)
        bt = set()
        backtrace(dp_mat, 0, len(line)-1)
        if DEBUG:
            print(line)
            print_mat(dp_mat)
            print(sorted(list(bt)))
            _ = input("bp")

        seg_l = list()
        for i in range(0, len(line)):
            seg_l.append(line[i])
            if i in bt and i != len(line)-1:
                seg_l.append(" ")
        seg_lines.append("".join(seg_l))

    write_txt(save_name, seg_lines)


def dp_cut(line):
    global front_ngram_VBE, back_ngram_VBE, front_ngram_avg, back_ngram_avg
    dp_mat = [[[None,None] for _ in range(len(line))] for _ in range(len(line))]
    for offset in range(0, len(line)):
        for i in range(0, len(line)):
            if i + offset >= len(line): continue
            cur = line[i:i+offset+1]
            length = offset + 1
            front = front_ngram_VBE[cur] if cur in front_ngram_VBE.keys() else -(1 << 31)
            front -= front_ngram_avg[length]
            back = back_ngram_VBE[cur] if cur in back_ngram_VBE.keys() else -(1 << 31)
            back -= front_ngram_avg[length]
            a_max = (front + back) * length
            seg_p = None
            if offset == 0:
                dp_mat[i][i][0] = a_max
                continue
            for j in range(i, offset):
                left = dp_mat[i][j][0] * (j-i+1)
                right = dp_mat[j+1][offset][0] * (offset-j)
                a = left + right
                if a > a_max:
                    a_max = a
                    seg_p = j
            dp_mat[i][i+offset][0], dp_mat[i][i+offset][1] = a_max, seg_p
    return dp_mat


def backtrace(dp_mat, lo, hi):
    global bt
    if lo > hi: return
    seg_p = dp_mat[lo][hi][1]
    if seg_p is None:
        bt.add(hi)
        return
    backtrace(dp_mat, lo, seg_p)
    backtrace(dp_mat, seg_p+1, hi)


def print_mat(mat):
    for line in mat: print(line)
