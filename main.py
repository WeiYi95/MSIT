# -*- coding:utf-8 -*-
# @Author: Wei Yi

import utils
from multiStage_Incremental_Training import multiStage_Incremental_Training


SIT = multiStage_Incremental_Training()

if __name__ == "__main__":
    utils.init_BE()
    SIT.train()
