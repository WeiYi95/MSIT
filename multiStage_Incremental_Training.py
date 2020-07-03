# -*- coding:utf-8 -*-
# @Author: Wei Yi

import utils
import os
from config import Config
from BERT_TRAIN import seg_train


class multiStage_Incremental_Training:
    def __init__(self):
        pass

    def one_iter(self, filename_list, iters, is_init=False):
        utils.produce_sampling_file(len(filename_list))
        utils.tri_select_sentences(filename_list, is_init)
        utils.labeled_txt_to_tf()

        if utils.read_from_pkl("data/new_sents.pkl") < Config.min_sent_increase:
            return
        print("Select", utils.read_from_pkl("data/new_sents.pkl"), ", total", utils.read_from_pkl("data/selected_sents.pkl"))
        utils.del_checkpoint()
        self.auto_train(iters)
        utils.mv_checkpoint()

    def train(self):
        iters = 0
        self.one_iter(["temp/sample_0.txt", "temp/sample_1.txt", "temp/sample_2.txt"], iters, True)

        utils.predict("data/data.txt", "data/pred.tf_record", "temp", as_text=True)
        iters += 1
        while iters <= Config.tot_stage:
            self.one_iter(["temp/sample_0.txt", "temp/sample_2.txt", "temp/predict.txt"], iters)

            if utils.read_from_pkl("data/new_sents.pkl") < Config.min_sent_increase:
                break

            utils.predict("data/data.txt", "data/pred.tf_record", "temp", as_text=True)
            iters += 1

        utils.copy_good_ckpt()
        utils.predict("data/data.txt", "data/train.tf_record", "temp", as_text=True)
        utils.append_pred()
        utils.save_as_pkl("data/selected_sents.pkl", len(utils.read_txt_file("data/train.txt")))
        utils.labeled_txt_to_tf()
        utils.del_checkpoint()
        self.auto_train(iters)
        utils.mv_checkpoint()
    
    def auto_train(self, cur_idx):
        tot_iter = Config.min_iters
        while True:
            utils.save_as_pkl("data/train_iter.pkl", tot_iter)
            seg_train()
            loss = utils.get_loss()
            if loss < Config.max_allowed_loss:
                Config.last_ckpt = cur_idx
                with open("best_ckpt.txt", "a", encoding="utf-8") as file:
                    file.write("best checkpoint: stage_" + str(Config.last_ckpt) + "_ckpt\n")
                break
            else:
                print("Restart training ...", utils.read_from_pkl("data/selected_sents.pkl"), "examples with train iters of", tot_iter, "failed ... Abnormal loss:", loss)
                tot_iter += 5
                if tot_iter > Config.max_iters:
                    break
                utils.del_checkpoint()
        utils.save_ckpt()
