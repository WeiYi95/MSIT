# -*- coding:utf-8 -*-
# @Author: Wei Yi

import pickle


def save_as_pkl(filename, data):
    output_pkl = open(filename, 'wb')
    pickle.dump(data, output_pkl)
    output_pkl.close()


def read_from_pkl(filename):
    pkl = open(filename, 'rb')
    data = pickle.load(pkl)
    pkl.close()
    return data


def read_txt(filename):
    with open(filename, 'r', encoding="utf-8") as file:
        content = file.read().split('\n')
    return content
