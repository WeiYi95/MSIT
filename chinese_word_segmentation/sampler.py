# -*- coding:utf-8 -*-
# @Author: Wei Yi

from chinese_word_segmentation.config import Config
from chinese_word_segmentation.DataLoader import DataLoader
import math


debug = Config.debug_GibbsSampler


class GibbsSampler:
    def __init__(self):
        self.data_loader = DataLoader()

    def sample(self, freq_dict, textfile, max_iters, save=True,  save_name=None):
        self.data_loader.update_freq_dict(filename=textfile, freq_dict=freq_dict)
        all_sent_tag = self.data_loader.get_all_sent_tag()
        unchange_idx = set()
        for iter in range(max_iters):
            for data_idx, (sent, tag) in enumerate(all_sent_tag):
                if data_idx in unchange_idx:
                    continue
                #if len("".join(sent[1:-1])) <= 12:
                #    unchange_idx.add(data_idx)
                #    continue
                if debug:
                    if data_idx != len(all_sent_tag)-1: continue

                is_change = False
                idx = 1

                while idx < len(sent) - 1:
                    try:  # There is a bug in the following code block
                        if debug:
                            print("------Before sampling------")
                            print(sent)
                            print(tag)
                            print(idx, len(sent))
                        cur_word = None
                        left_context = (None, None)
                        right_context = (None, None)
                        cur_len = len(sent[idx])
                        if idx != len(sent) - 1:
                            cur_word = [(sent[idx], tag[idx]), (sent[idx+1], tag[idx+1])]
                        else:
                            cur_word = [(sent[idx], tag[idx])]
                        if idx+2 < len(sent):
                            right_context = (sent[idx + 2], tag[idx + 2])
                        if idx != 0:
                            left_context = (sent[idx-1], tag[idx-1])

                        if left_context[0] == 'B' or right_context[0] == 'S':
                            idx += 1
                            continue

                        if len(cur_word) == 2:
                            freq_dict.undo_one_step(cur_word=cur_word[0], left_context=left_context, right_context=cur_word[1])
                            freq_dict.undo_one_step(cur_word=cur_word[1], left_context=(None, None), right_context=right_context)
                        else:
                            freq_dict.undo_one_step(cur_word=cur_word[0], left_context=left_context, right_context=right_context)

                        subpart, tags = self.segment_subpart(freq_dict, cur_word, left_context, right_context)
                        left_word = [None] if left_context[0] is None else [left_context[0]]
                        right_word = [None] if right_context[0] is None else [right_context[0]]
                        left_tag = [None] if left_context[1] is None else [left_context[1]]
                        right_tag = [None] if right_context[1] is None else [right_context[1]]

                        if len(subpart[0]) != cur_len:
                            is_change = True

                        if debug:
                            print(left_word + subpart + right_word)
                            print(left_tag + tags + right_tag)
                        freq_dict.update_one_step(left_word + subpart + right_word,
                                         left_tag + tags + right_tag)
                        if idx != len(sent) - 1:
                            sent = sent[:idx] + subpart + sent[idx+2:]
                            tag = tag[:idx] + tags + tag[idx+2:]
                        else:
                            sent = sent[:idx] + subpart
                            tag = tag[:idx] + tags
                        all_sent_tag[data_idx] = (sent, tag)
                        if debug:
                            print("------After sampling------")
                            print(sent)
                            print(tag)
                            i = input("---")

                        idx += 1

                        if is_change is False:
                            unchange_idx.add(data_idx)
                    except:
                        break
        if save:
            self.save_text(save_name, [sent for sent, _ in all_sent_tag])


    def segment_subpart(self, freq_dict, cur_word, left_context, right_context):
        # return segmentation with corresponding tags as a list
        unseg_str = cur_word[0][0] if len(cur_word) == 1 else cur_word[0][0] + cur_word[1][0]
        idx = self.get_best_segmentation(freq_dict, unseg_str, left_context, right_context)
        fir_word = unseg_str[:idx]
        sec_word = unseg_str[idx:]
        sent = None
        if sec_word != '':
            sent = [fir_word, sec_word]
        else:
            sent = [fir_word]
        _, tags = self.data_loader.tag_sent(sent)

        if debug:
            print(sent, tags)

        return sent, tags

    def get_best_segmentation(self, freq_dict, unseg_str, left_context, right_context):
        # return pos start at 1 instead of 0!
        best_prob = -1e10
        best_idx = 0
        for idx in range(1, len(unseg_str)+1):
            fir_word = unseg_str[:idx]
            sec_word = unseg_str[idx:]
            if debug:
                print(fir_word, sec_word)
            _, tags = self.data_loader.tag_sent([fir_word, sec_word])
            hmm_prob1 = self.HMM_prob(freq_dict, fir_word, tags[0])
            hmm_prob2 = -999
            if sec_word != '':
                hmm_prob2 = self.HMM_prob(freq_dict, sec_word, tags[1], fir2sec=[tags[0][-1], tags[1][0]])
            hmm_prob = hmm_prob1 + hmm_prob2

            crp_prob1 = -999
            if left_context[0] is not None:
                crp_prob1 = self.CRP_prob(freq_dict, fir_word, context=left_context[0])
            crp_prob2 = -999
            crp_prob3 = -999
            if sec_word != '':
                crp_prob2 = self.CRP_prob(freq_dict, sec_word, context=fir_word)
                if right_context[0] is not None:
                    crp_prob3 = self.CRP_prob(freq_dict, right_context[0], context=sec_word)
            elif right_context[0] is not None:
                crp_prob2 = self.CRP_prob(freq_dict, right_context[0], context=fir_word)
            crp_prob = crp_prob1 + crp_prob2 + crp_prob3

            tot_log_prob = hmm_prob + crp_prob
            if tot_log_prob > best_prob:
                best_prob = tot_log_prob
                best_idx = idx

            if debug:
                print("hmm prob1: %f" % hmm_prob1)
                print("hmm prob2: %f" % hmm_prob)
                print("crp prob1: %f" % crp_prob1)
                print("crp prob2: %f" % crp_prob2)
                print("crp prob3: %f" % crp_prob3)
                print("tot prob: %f" % tot_log_prob)
                print("current best idx: %d" % best_idx)
                print("current best score: %f" % best_prob)
                print()
        if debug:
            print("final best idx: %d" % best_idx)
        return best_idx

    def HMM_prob(self, freq_dict, word, tags, fir2sec=None):
        tot_porb = 0
        for idx in range(len(tags)-1):
            cur_tag = tags[idx]
            next_tag = tags[idx+1]
            cur_char = word[idx]

            n_ti_1_ti = 0.0 if cur_tag not in freq_dict.trans_freq.keys() else freq_dict.trans_freq[cur_tag][0][next_tag]
            n_ti_dot = 0.0 if cur_tag not in freq_dict.trans_freq.keys() else freq_dict.trans_freq[cur_tag][1]
            trans_prob = math.log(n_ti_1_ti + Config.theta + Config.esp) - \
                math.log(n_ti_dot + Config.tag_size*Config.theta + Config.esp)

            n_ti_ci = 0.0 if cur_tag not in freq_dict.emit_freq.keys() else freq_dict.emit_freq[cur_tag][0][cur_char]
            n_ti_cdot = 0.0 if cur_tag not in freq_dict.emit_freq.keys() else freq_dict.emit_freq[cur_tag][1]
            emit_prob = math.log(n_ti_ci + Config.sigma + Config.esp) - \
                math.log(n_ti_cdot + Config.vocab_size*Config.sigma + Config.esp)
            tot_porb += trans_prob
            tot_porb += emit_prob

        cur_tag = tags[-1]
        cur_char = word[-1]
        n_ti_ci = 0.0 if cur_tag not in freq_dict.emit_freq.keys() else freq_dict.emit_freq[cur_tag][0][cur_char]
        n_ti_cdot = 0.0 if cur_tag not in freq_dict.emit_freq.keys() else freq_dict.emit_freq[cur_tag][1]
        emit_prob = math.log(n_ti_ci + Config.sigma + Config.esp) - \
                    math.log(n_ti_cdot + Config.vocab_size * Config.sigma + Config.esp)
        tot_porb += emit_prob

        # deal with transition from first word to second word
        if fir2sec is not None:
            cur_tag = fir2sec[0]
            next_tag = fir2sec[1]
            n_ti_1_ti = 0.0 if cur_tag not in freq_dict.trans_freq.keys() else freq_dict.trans_freq[cur_tag][0][
                next_tag]
            n_ti_dot = 0.0 if cur_tag not in freq_dict.trans_freq.keys() else freq_dict.trans_freq[cur_tag][1]
            trans_prob = math.log(n_ti_1_ti + Config.theta + Config.esp) - \
                         math.log(n_ti_dot + Config.tag_size * Config.theta + Config.esp)
            tot_porb += trans_prob

        return tot_porb

    def CRP_prob(self, freq_dict, word, context):
        #cnt = 0 if word not in freq_dict.word_freq.keys() else freq_dict.word_freq[word]
        #word_cnt = max(0, cnt-1)
        word_cnt = max(0, len(word)-1)
        char_prob = 1
        for c in word:
            if c not in freq_dict.character_wordfreq.keys():
                char_prob = 0
                break
            elif word not in freq_dict.character_wordfreq[c][0].keys():
                char_prob = 0
                break
            else:

                if freq_dict.character_wordfreq[c][1] == 0:
                    char_prob = 0
                    break
                char_prob *= (freq_dict.character_wordfreq[c][0][word] / freq_dict.character_wordfreq[c][1])
        h_wi = math.pow(1-Config.p_s, word_cnt) * Config.p_s * char_prob
        p_wi_h = (freq_dict.rstrnt.word_tablecnt[word] + Config.alpha*h_wi + Config.esp) / \
            (freq_dict.rstrnt.tot_table + Config.alpha + Config.esp)
        n_wi_wi_1 = 0.0 if word not in freq_dict.bigram_contextfreq.keys() else freq_dict.bigram_contextfreq[word][0][context]
        n_wi_dot = 0.0 if word not in freq_dict.bigram_contextfreq.keys() else freq_dict.bigram_contextfreq[word][1]
        prob = math.log(n_wi_wi_1 + Config.alpha_one*p_wi_h + Config.esp) - \
            math.log(n_wi_dot + Config.alpha_one + Config.esp)
        if debug:
            print("n_wi_wi_1 %f" % n_wi_wi_1)
            print("n_wi_dot %f" % n_wi_dot)
            print("h_wi %f" % h_wi)
            print("p_wi_h %f" % p_wi_h)
            print("p %f" % ((n_wi_wi_1 + Config.alpha_one*p_wi_h + Config.esp)/(n_wi_dot + Config.alpha_one + Config.esp)))
        return prob

    def update_one_step(self, freq_dict, cur_word, left_context, right_context):
        sent = [left_context[0]]
        tags = [left_context[1]]
        for tup in cur_word:
            sent.append(tup[0])
            tags.append(tup[1])
        sent.append(right_context[0])
        tags.append(right_context[1])
        freq_dict.update_one_step(sent, tags)

    def save_text(self, filename, sent):
        with open(filename, 'w', encoding="utf-8") as file:
            for line in sent:
                file.write(" ".join(line[1:-1]) + '\n')
