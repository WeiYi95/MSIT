# -*- coding:utf-8 -*-
# @Author: Wei Yi

import os
import random
import pickle
import BE
import shutil
from config import Config
from chinese_word_segmentation.sampler import GibbsSampler
from chinese_word_segmentation.FreqDict import FreqDict
from chinese_word_segmentation.DataLoader import DataLoader
from produce_tf import produce_tfrecord
from to_tfdata import produce_tfrecord as pt
from BERT_PRED import BERT_PRED

j2f_dict = None
f2j_dict = None

def j2f(sents):
    global j2f_dict
    if j2f_dict is None:
        j2f_dict = dict()
        with open("/yjs/euphoria/wy/electra/dataset/multistage/data.txt", "r", encoding="utf-8") as file:
            j = file.read().split("\n")
        with open("/yjs/euphoria/wy/electra/dataset/multistage/data_ft.txt", "r", encoding="utf-8") as file:
            f = file.read().split("\n")
        for i in range(len(j)):
            #assert len(j[i]) == len(f[i])
            j2f_dict[j[i]] = f[i]
    f_sents = list()
    for idx, s in enumerate(sents):
        jt = s.split(" ")
        ft = j2f_dict["".join(jt)]
        p = 0
        sent = list()
        for w in jt:
            word = list()
            for i,c in enumerate(w):
                word.append(ft[p])
                p += 1
            sent.append("".join(word))
        #assert "".join(sent) == ft
        f_sents.append(" ".join(sent))
    return f_sents


def f2j(sents):
    global f2j_dict
    if f2j_dict is None:
        f2j_dict = dict()
        with open("/yjs/euphoria/wy/electra/dataset/multistage/data.txt", "r", encoding="utf-8") as file:
            j = file.read().split("\n")
        with open("/yjs/euphoria/wy/electra/dataset/multistage/data_ft.txt", "r", encoding="utf-8") as file:
            f = file.read().split("\n")
        for i in range(len(j)):
            #assert len(j[i]) == len(f[i])
            f2j_dict[f[i]] = j[i]
    j_sents = list()
    for idx, s in enumerate(sents):
        ft = s.split(" ")
        try:
            jt = f2j_dict["".join(ft)]
        except:
            continue
        p = 0
        sent = list()
        for w in ft:
            word = list()
            for i,c in enumerate(w):
                word.append(jt[p])
                p += 1
            sent.append("".join(word))
        #assert "".join(sent) == ft
        j_sents.append(" ".join(sent))
    return j_sents


def read_from_pkl(filename):
    pkl = open(filename, 'rb')
    data = pickle.load(pkl)
    pkl.close()
    return data


def save_as_pkl(filename, data):
    output_pkl = open(filename, 'wb')
    pickle.dump(data, output_pkl)
    output_pkl.close()


def check_dir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


if os.path.exists("data/train_step.pkl"):
    os.remove("data/train_step.pkl")
if os.path.exists("data/fq.pkl"):
    os.remove("data/fq.pkl")
check_dir("data")
check_dir("temp")
check_dir("tfdata")
check_dir("output")
INPUT_FILE = "data/data.txt"
TEMP_FILE = "temp/temp.txt"
OUTPUT_FILE = "data/train.txt"
REM_FILE = "data/data.txt"
TRAIN_FILE = "data/msit.txt"
if os.path.exists(OUTPUT_FILE):
    os.remove(OUTPUT_FILE)
if os.path.exists(TRAIN_FILE):
    os.remove(TRAIN_FILE)
CUR_STAGE = 0


def read_txt_file(filename):
    with open(filename, 'r', encoding="utf-8") as file:
        return file.read().split('\n')


def save_txt_file(filename, sents):
    with open(filename, 'w', encoding="utf-8") as file:
        for idx, sent in enumerate(sents):
            if idx != len(sents) - 1:
                file.write(sent + '\n')
            else:
                file.write(sent)


def random_cutting(sents, thres=6):
    result = []
    if thres in set(["BE", "nVBE", "jieba"]):
        s2s = read_from_pkl("seg_" + thres + ".pkl")
        for sent in sents:
            if sent not in s2s.keys():
                result.append(sent)
                continue
            result.append(s2s[sent])
    return result


def sampling(input_filename, output_filename):
    gs = GibbsSampler()
    fq = FreqDict()
    if os.path.exists("data/fq.pkl"):
        fq = FreqDict(load=True, pkl_name="data/fq.pkl")
    gs.sample(freq_dict=fq, textfile=input_filename, max_iters=100, save=True, save_name=output_filename)


def produce_sampling_file(output_cnt):
    global INPUT_FILE, TEMP_FILE
    seg_type = ["BE", "nVBE", "jieba"]
    for cnt in range(output_cnt):
        sampling_filename = "temp/sample_" + str(cnt) + ".txt"
        sents = read_txt_file(INPUT_FILE)
        sents = random_cutting(sents, thres=seg_type[cnt])
        save_txt_file(TEMP_FILE, sents=sents)
        sampling(input_filename=TEMP_FILE, output_filename=sampling_filename)


def check_word_len(sent):
    for w in sent.split(' '):
        if len(w) > 1:
            return False
    return True


def append_pred():
    global OUTPUT_FILE, TRAIN_FILE
    has_output = os.path.exists(TRAIN_FILE)
    selected_cnt = 0
    if has_output:
        with open(TRAIN_FILE, 'r', encoding="utf-8") as file:
            selected_cnt += len(file.read().split("\n"))
    with open("temp/predict.txt", 'r', encoding="utf-8") as file:
        with open(TRAIN_FILE, 'a', encoding="utf-8") as afile:
            lines = [line for line in file.read().split("\n") if len(line) > 0]
            for i, line in enumerate(lines):
                afile.write(line)
                if i != len(lines) - 1:
                    afile.write("\n")
    with open(TRAIN_FILE, 'r', encoding="utf-8") as file:
        with open(OUTPUT_FILE, 'w', encoding="utf-8") as rfile:
            lines = list()
            for line in file.read().split("\n"):
                if len(line) > 0:
                    lines.append(line)
            for i, line in enumerate(lines):
                rfile.write(line)
                if i != len(lines) - 1:
                    rfile.write("\n")


def tri_select_sentences(filename_list, is_init=False):
    global OUTPUT_FILE, REM_FILE, TRAIN_FILE
    sents_one = read_txt_file(filename_list[0])
    sents_two = read_txt_file(filename_list[1])
    sents_thr = read_txt_file(filename_list[2])
    selected_cnt = 0
    new_sent_cnt = 0
    has_output = os.path.exists(TRAIN_FILE)
    if has_output:
        with open(TRAIN_FILE, 'r', encoding="utf-8") as file:
            selected_cnt += len(file.read().split("\n"))
    with open(TRAIN_FILE, 'a', encoding="utf-8") as file:
        with open(REM_FILE, 'w', encoding="utf-8") as rfile:
            for idx in range(len(sents_one)):
                is_last = True if idx == len(sents_one) - 1 else False
                if sents_one[idx] == "":
                    continue
                agree = 0
                if sents_one[idx] == sents_two[idx]:
                    agree += 1
                if sents_one[idx] == sents_thr[idx]:
                    agree += 1
                if sents_two[idx] == sents_thr[idx]:
                    agree += 1
                is_all_one = check_word_len(sents_one[idx])
                if is_all_one:
                    is_all_one = check_word_len(sents_two[idx])
                if is_all_one:
                    is_all_one = check_word_len(sents_thr[idx])
                if is_init:
                    is_all_one = False
                if agree == 3:
                    # if agree == 3 and not is_all_one:
                    selected_cnt += 1
                    new_sent_cnt += 1
                    file.write(sents_one[idx] + '\n')
                else:
                    if is_last:
                        rfile.write("".join(sents_one[idx].split(' ')) + '\n')
                    else:
                        rfile.write("".join(sents_one[idx].split(' ')) + '\n')
    with open(TRAIN_FILE, 'r', encoding="utf-8") as file:
        with open(OUTPUT_FILE, 'w', encoding="utf-8") as rfile:
            lines = list()
            for line in file.read().split("\n"):
                if len(line) > 0:
                    lines.append(line)
            for i, line in enumerate(lines):
                rfile.write(line)
                if i != len(lines) - 1:
                    rfile.write("\n")

    fq = FreqDict()
    if os.path.exists("data/fq.pkl"):
        fq = FreqDict(load=True, pkl_name="data/fq.pkl")
    with open("temp/fq.txt", 'w', encoding="utf-8") as file:
        for idx in range(len(sents_one)):
            file.write(sents_one[idx] + '\n')
    data_loader = DataLoader()
    fq = data_loader.update_freq_dict(filename="temp/fq.txt", freq_dict=fq, need_return=True)
    fq.save("data/fq.pkl")
    save_as_pkl("data/selected_sents.pkl", selected_cnt)
    save_as_pkl("data/new_sents.pkl", new_sent_cnt)


def to_label(line):
    words = line.split(' ')
    label = []
    for word in words:
        label.append("[BOS]")
        rem = len(word) - 1
        for _ in range(rem):
            label.append("[IOS]")
    return "".join(words), " ".join(label)


def to_seg_data(OUTPUT_FILE, out_filename=None):
    if out_filename is None:
        out_filename = OUTPUT_FILE
    with open(OUTPUT_FILE, "r", encoding="utf-8") as file:
        lines = file.read().split("\n")
    f_lines = j2f(lines)
    with open("f_data.txt", "w", encoding="utf-8") as file:
        for i, line in enumerate(f_lines):
            file.write(line)
            if i != len(lines) - 1: file.write("\n")
    seg_lines = []
    with open("f_data.txt", 'r', encoding="utf-8") as file:
        lines = file.read().split('\n')
        for line in lines:
            line, label = to_label(line)
            if len(line) > 22:
                continue
            seg_lines.append(line + '|' + label)
    with open(out_filename, 'w', encoding="utf-8") as file:
        for seg in seg_lines:
            file.write(seg + '\n')


def labeled_txt_to_tf():
    global OUTPUT_FILE
    to_seg_data(OUTPUT_FILE)
    produce_tfrecord()


def del_checkpoint():
    filename_list = os.listdir("output/")
    for filename in filename_list:
        need_del = True
        if filename.find("model.ckpt") != -1 or filename == "checkpoint":
            need_del = True
        if need_del:
            os.remove("output/" + filename)


def mv_checkpoint(outdir="output/"):
    filename_list = os.listdir("output")
    need_del_name_list = ["events.out", "model.ckpt-0", "graph.pbtxt"]
    for filename in filename_list:
        need_del = False
        for name in need_del_name_list:
            if filename.find(name) != -1:
                need_del = True
                break
        if need_del:
            os.remove("output/" + filename)
            continue

        if filename.find(".data") != -1:
            new_name = "model.ckpt.data-00000-of-00001"
            os.rename("output/" + filename, outdir + new_name)

        if filename.find(".meta") != -1:
            new_name = "model.ckpt.meta"
            os.rename("output/" + filename, outdir + new_name)

        if filename.find("index") != -1:
            new_name = "model.ckpt.index"
            os.rename("output/" + filename, outdir + new_name)

        if filename.find("checkpoint") != -1:
            new_name = "checkpoint"
            os.rename("output/" + filename, outdir + new_name)


def save_ckpt():
    global CUR_STAGE
    save_ckpt_dir = "stage_" + str(CUR_STAGE) + "_ckpt/"
    if not os.path.exists(save_ckpt_dir):
        os.mkdir(save_ckpt_dir)
    CUR_STAGE += 1
    filename_list = os.listdir("output")
    need_del_name_list = ["events.out", "model.ckpt-0", "graph.pbtxt"]
    for filename in filename_list:
        need_del = False
        for name in need_del_name_list:
            if filename.find(name) != -1:
                need_del = True
                break
        if need_del:
            os.remove("output/" + filename)
            continue

        if filename.find(".data") != -1:
            new_name = "model.ckpt.data-00000-of-00001"
            os.rename("output/" + filename, save_ckpt_dir + new_name)

        if filename.find(".meta") != -1:
            new_name = "model.ckpt.meta"
            os.rename("output/" + filename, save_ckpt_dir + new_name)

        if filename.find("index") != -1:
            new_name = "model.ckpt.index"
            os.rename("output/" + filename, save_ckpt_dir + new_name)

        if filename.find("checkpoint") != -1:
            new_name = "checkpoint"
            os.rename("output/" + filename, save_ckpt_dir + new_name)
    shutil.rmtree("output/")
    shutil.copytree(save_ckpt_dir, "output/")


def copy_good_ckpt():
    good_ckpt_dir = "stage_" + str(Config.last_ckpt) + "_ckpt/"
    shutil.rmtree("output/")
    shutil.copytree(good_ckpt_dir, "output/")



def label2text(text, label):
    res = []
    label = label.split(' ')[1:-1]
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


def from_label_to_text(text_file, label_file, output_file):
    with open(text_file, 'r', encoding="utf-8") as file:
        gt = file.read().split('\n')

    with open(label_file, 'r', encoding="utf-8") as file:
        pred = file.read().split('\n')
    p = list()
    with open(output_file, 'w', encoding="utf-8") as file:
        for i in range(len(gt)):
            if gt[i] == "":
                continue
            text = "".join(gt[i].split(' '))
            if len(text) == 0:
                continue
            pred_label = pred[i]
            prediction = label2text(text, pred_label)
            p.append(prediction)
        j_p = f2j(p)
        for i in range(len(j_p)):
            if i == len(gt) - 1:
                file.write(j_p[i])
            else:
                file.write(j_p[i] + '\n')


def predict(pred_filename, output_filename, output_dir, as_text):
    with open(pred_filename, "r", encoding="utf-8") as file:
        lines = file.read().split("\n")
    f_lines = j2f(lines)
    with open("f_data.txt", "w", encoding="utf-8") as file:
        for i, line in enumerate(f_lines):
            file.write(line)
            if i != len(lines) - 1: file.write("\n")
    to_seg_data("f_data.txt", "temp/temp.txt")
    save_as_pkl("pred_filename.pkl", "temp/temp.txt")
    save_as_pkl("output_filename.pkl", output_filename)
    save_as_pkl("output_dir.pkl", output_dir)
    pt()
    BERT_PRED()
    if as_text:
        from_label_to_text(pred_filename, output_dir + "/pred.txt", "temp/predict.txt")


def get_loss():
    with open("log", 'r') as file:
        lines = file.read().split('\n')
        for idx in range(len(lines) - 1, -1, -1):
            if lines[idx].find("Loss for final step") != -1:
                return float(lines[idx].split(' ')[-1][:-1])


def init_BE():
    with open(Config.data_fn, 'r', encoding="utf-8") as file:
        lines = [line for line in file.read().split("\n") if len(line) > 0]
    with open("data/data.txt", 'w', encoding="utf-8") as file:
        for i, line in enumerate(lines):
            file.write(line)
            if i != len(lines) - 1:
                file.write("\n")
    BE.init(corpus="", fn="data/data.txt", redo=False, use_jieba=False)
