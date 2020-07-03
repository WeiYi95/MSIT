# -*- coding:utf-8 -*-
# @Author: Wei Yi

import chinese_word_segmentation.utils as utils
from collections import defaultdict
from chinese_word_segmentation.config import Config


debug = Config.debug_FreqDict


class Restaurant:
    def __init__(self):
        self.unigram_rstrnt = defaultdict(int)
        self.bigram_rstrnt = {}
        self.word_tablecnt = defaultdict(int)
        self.tot_table = 0

    def show(self):
        print(self.unigram_rstrnt)
        print(self.bigram_rstrnt)
        print(self.word_tablecnt)
        print(self.tot_table)


class FreqDict:
    def __init__(self, load=False, pkl_name=None):
        self.character_wordfreq = {}
        # value[0]: word-freq pair
        # value[1]: total words count

        self.word_freq = defaultdict(int)
        # word-freq pair

        self.bigram_contextfreq = {}
        # key: context
        # value[0]: context-freq pair
        # value[1]: context freq

        self.emit_freq = {}
        # key: type
        # value[0]: character-freq pair
        # value[1]: total emission count

        self.trans_freq = {}
        # key: type
        # value[0]: type-freq pair
        # value[1]: total transition count

        self.rstrnt = Restaurant()
        # unigram_rstrnt: uni-gram restaurant
            # key: uni-gram word
            # value: #coustomer
        # bigram_rstrnt: word restaurant
            # key: word
            # value[0]: #coustomer
        # word_tablecnt: word-table pair
            # key: word
            # value: table associated with the word
        # tot_table: total table count

        if load:
            assert pkl_name is not None, "Need provide pkl name if load is True"
            self.load(pkl_name)

    def update(self, sent_tags):
        """
        :param sent_tags: tuple(sent, tag) tag is {SBME}
                    这 是 关于 迪利克雷过程 的 内容
                    S  S  B E  B M M M M E  S  B E
                    (["这", "是", "关于", "迪利克雷过程", "的", "内容"],
                     ["S",  "S",  "BE",   "BMMMME",       "S",  "BE"])
        :return:
        """
        sent, tags = sent_tags
        length = len(sent)
        for idx in range(length):
            cur_word = sent[idx]
            cur_tags = tags[idx]

            self.word_freq[cur_word] += 1

            if idx != length - 1:
                next_word = sent[idx+1]
                next_tags = tags[idx+1]

                self.update_dict_with_list_value(self.bigram_contextfreq, cur_word, next_word)

                # update HMM parameters last tag of current word transition to first tag of next word
                self.update_dict_with_list_value(self.trans_freq, cur_tags[-1], next_tags[0])

                self.update_CRP(cur_word, next_word)

            # update HMM parameters
            for pos in range(len(cur_word)):
                cur_char = cur_word[pos]
                cur_tag = cur_tags[pos]
                if pos != len(cur_word) - 1:
                    next_tag = cur_tags[pos+1]

                self.update_dict_with_list_value(self.character_wordfreq, cur_char, cur_word)

                self.update_dict_with_list_value(self.emit_freq, cur_tag, cur_char)
                if pos != len(cur_word) - 1:
                    self.update_dict_with_list_value(self.trans_freq, cur_tag, next_tag)

        if debug:
            self.show_all()

    def update_CRP(self, cur_word, next_word):
        need_unigram_table = False

        if cur_word not in self.rstrnt.bigram_rstrnt.keys():
            self.rstrnt.bigram_rstrnt[cur_word] = defaultdict(int)  # open a new restaurant

        if next_word not in self.rstrnt.bigram_rstrnt[cur_word].keys():
            need_unigram_table = True  # form a new table

            self.rstrnt.tot_table += 1
            self.rstrnt.word_tablecnt[next_word] += 1

        self.rstrnt.bigram_rstrnt[cur_word][next_word] += 1  # next word sit into the table

        if need_unigram_table:
            if next_word not in self.rstrnt.unigram_rstrnt.keys():
                self.rstrnt.tot_table += 1
                self.rstrnt.word_tablecnt[next_word] += 1
            self.rstrnt.unigram_rstrnt[next_word] += 1

    def undo_one_step(self, cur_word, left_context, right_context):
        self.undo_HMM_CRP(left_context, cur_word, right_context)

    def undo_HMM_CRP(self, left_context, cur_word, right_context):
        # undo HMM parameters
        if debug:
            print(left_context)
            print(cur_word)
            print(right_context)
        tags = cur_word[1]
        words = cur_word[0]
        last_type = tags[0]
        self.word_freq[words] -= 1
        if left_context[0] is not None:
            last_type = left_context[1][-1]  # have left context
        else:
            tags = tags[1:]  # start at the second tag of current word
            self.emit_freq[last_type][0][words[0]] -= 1
            self.emit_freq[last_type][1] -= 1
            words = words[1:]

        for idx, cur_type in enumerate(tags):
            self.trans_freq[last_type][0][cur_type] -= 1
            self.trans_freq[last_type][1] -= 1
            self.emit_freq[cur_type][0][words[idx]] -= 1
            self.emit_freq[cur_type][1] -= 1
            last_type = cur_type

        if right_context[0] is not None:
            cur_type = right_context[1][0]
            self.trans_freq[last_type][0][cur_type] -= 1
            self.trans_freq[last_type][1] -= 1

        # undo CRP parameters
        word = cur_word[0]
        for c in word:
            self.character_wordfreq[c][0][word] -= 1
            self.character_wordfreq[c][1] -= 1

        if left_context[0] is not None:
            left_context = left_context[0]
            self.bigram_contextfreq[left_context][0][word] -= 1
            self.bigram_contextfreq[left_context][1] -= 1

            if self.rstrnt.bigram_rstrnt[left_context][word] > 1:
                self.rstrnt.bigram_rstrnt[left_context][word] -= 1
            elif self.rstrnt.bigram_rstrnt[left_context][word] == 1:
                self.rstrnt.bigram_rstrnt[left_context][word] -= 1
                self.rstrnt.unigram_rstrnt[word] -= 1
                self.rstrnt.word_tablecnt[word] -= 1
                self.rstrnt.tot_table -= 1
                del self.rstrnt.bigram_rstrnt[left_context][word]
                if self.rstrnt.unigram_rstrnt[word] == 0:
                    del self.rstrnt.unigram_rstrnt[word]
                    self.rstrnt.word_tablecnt[word] -= 1
                    self.rstrnt.tot_table -= 1

        if right_context[0] is not None:
            right_context = right_context[0]
            self.bigram_contextfreq[word][0][right_context] -= 1
            self.bigram_contextfreq[word][1] -= 1

            if self.rstrnt.bigram_rstrnt[word][right_context] > 1:
                self.rstrnt.bigram_rstrnt[word][right_context] -= 1
            elif self.rstrnt.bigram_rstrnt[word][right_context] == 1:
                self.rstrnt.bigram_rstrnt[word][right_context] -= 1
                self.rstrnt.unigram_rstrnt[right_context] -= 1

                self.rstrnt.word_tablecnt[right_context] -= 1
                self.rstrnt.tot_table -= 1
                del self.rstrnt.bigram_rstrnt[word][right_context]
                if self.rstrnt.unigram_rstrnt[right_context] == 0:
                    del self.rstrnt.unigram_rstrnt[right_context]
                    self.rstrnt.word_tablecnt[right_context] -= 1
                    self.rstrnt.tot_table -= 1

        if debug:
            self.show_all()

    def update_one_step(self, sent, tags):
        tot = len(sent) - 2
        for idx in range(1, tot+1):
            cur_word = sent[idx]
            cur_tags = tags[idx]

            self.word_freq[cur_word] += 1

            if sent[idx+1] is not None:
                next_word = sent[idx+1]
                next_tags = tags[idx+1]

                self.update_dict_with_list_value(self.bigram_contextfreq, cur_word, next_word)

                # update HMM parameters last tag of current word transition to first tag of next word
                self.update_dict_with_list_value(self.trans_freq, cur_tags[-1], next_tags[0])

                self.update_CRP(cur_word, next_word)

            # update HMM parameters
            for pos in range(len(cur_word)):
                cur_char = cur_word[pos]
                cur_tag = cur_tags[pos]
                if pos != len(cur_word) - 1:
                    next_tag = cur_tags[pos+1]

                self.update_dict_with_list_value(self.character_wordfreq, cur_char, cur_word)

                self.update_dict_with_list_value(self.emit_freq, cur_tag, cur_char)
                if pos != len(cur_word) - 1:
                    self.update_dict_with_list_value(self.trans_freq, cur_tag, next_tag)

            if idx == 1 and sent[0] is not None:
                idx = 0
                cur_word = sent[idx]
                cur_tags = tags[idx]
                next_word = sent[idx + 1]
                next_tags = tags[idx + 1]

                self.update_dict_with_list_value(self.bigram_contextfreq, cur_word, next_word)

                # update HMM parameters last tag of current word transition to first tag of next word
                self.update_dict_with_list_value(self.trans_freq, cur_tags[-1], next_tags[0])

                self.update_CRP(cur_word, next_word)

        if debug:
            self.show_all()

    @staticmethod
    def update_dict_with_list_value(dict_name, key_name, value_key_name):
        if key_name not in dict_name.keys():
            dict_name[key_name] = [defaultdict(int), 0]
        dict_name[key_name][0][value_key_name] += 1
        dict_name[key_name][1] += 1

    def show_all(self):
        print(self.character_wordfreq)
        print(self.word_freq)
        print(self.bigram_contextfreq)
        print(self.emit_freq)
        print(self.trans_freq)
        self.rstrnt.show()


    def save(self, pkl_name):
        utils.save_as_pkl(pkl_name, [self.character_wordfreq, self.word_freq, self.bigram_contextfreq, self.emit_freq,
                                     self.trans_freq, self.rstrnt])

    def load(self, pkl_name):
        data = utils.read_from_pkl(pkl_name)
        self.character_wordfreq = data[0]
        self.word_freq = data[1]
        self.bigram_contextfreq = data[2]
        self.emit_freq = data[3]
        self.trans_freq = data[4]
        self.rstrnt = data[5]
