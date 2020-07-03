# -*- coding:utf-8 -*-
# @Author: Wei Yi

import chinese_word_segmentation.utils as utils
from chinese_word_segmentation.config import Config


debug = Config.debug_DataLoader


class DataLoader:
    def __init__(self):
        self.wordlist_taglist = None

    def update_freq_dict(self, filename, freq_dict, need_return=False):
        # each line in file should use space as word delimiter
        self.wordlist_taglist = []

        lines = utils.read_txt(filename)
        for line in lines:
            sent = self.sent2list("B " + line + " E")
            if sent is None:
                continue
            sent_tag = self.tag_sent(sent)
            if sent_tag is None:
                continue

            if debug:
                print(sent_tag[0])
                print(sent_tag[1])

            freq_dict.update(sent_tags=sent_tag)
            self.wordlist_taglist.append(sent_tag)
        if need_return:
            return freq_dict

    def tag_sent(self, sent):
        """
        :param sent: list of words
        :return: a tuple (sent -> list, tags -> list) to update FreqDict (see its update method)
        """
        tags = []
        word_list = []
        for word in sent:
            word = word.strip()
            if len(word) == 0:
                continue
            word_list.append(word)
            tag = self.tag_word(word)
            tags.append(tag)

        # consider the case where it only has one word, but it's a long word.
        # is it necessary to do a random segmentation so that it can be sampled.
        # or just leave it be?

        return word_list, tags

    # return all data for sampling
    def get_all_sent_tag(self):
        return self.wordlist_taglist

    # tag each with {SBME}
    @staticmethod
    def tag_word(word):
        word_len = len(word)
        if word_len == 1:
            return "S"
        elif word_len == 2:
            return "BE"
        elif word_len >= 3:
            m_len = word_len - 2
            return "B" + "M"*m_len + "E"

    # trim and make sure each sent is AT LEAST consist of TWO words. return a list of words
    # the input sentence uses space as word delimiter
    @staticmethod
    def sent2list(sent):
        sent = sent.strip().split(' ')
        if len(sent) <= 1:
            return None
        return sent
