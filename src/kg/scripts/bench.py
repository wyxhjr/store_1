import enum
from ftplib import all_errors
from numpy import outer
import time
import argparse
import itertools
import os
import subprocess
import datetime

from bench_util import RemoteExecute, ParallelSSH, Pnuke, GetHostName
import exp_config
from exp_config import ALL_SERVERS_INCLUDING_NOT_USED, LOG_PREFIX, PROJECT_PATH

from variables import *


def mount_master(hosts):
    pass


def config_each_server(hosts):
    ParallelSSH(hosts, f"sudo swapoff -a")


if GetHostName() == "node182":
    suffix = "A30"
else:
    suffix = "3090"


def main():
    exp_lists = []
    if suffix == "A30":
        # each = exp_config.ExpKGScalability()
        # each.SetLogDir(f'{LOG_PREFIX}/1015-KG-scale-large-{suffix}')
        # each.SetFilter(lambda config: config['dataset'] == 'wikikg2')
        # exp_lists.append(each)
        
        each = exp_config.ExpKGvsA30()
        each.SetLogDir(f'{LOG_PREFIX}/1017-KGvsA30-{suffix}')
        each.SetFilter(lambda config: config['cached_emb_type'] != 'KnownLocalCachedEmbedding')
        exp_lists.append(each)


        # each = exp_config.ExpRecPerf()
        # each.SetLogDir(f'{LOG_PREFIX}/1015-Rec-{suffix}')
        # each.SetFilter(lambda config: config['dataset'] == 'criteoTB')
        # exp_lists.append(each)
        
        return exp_lists

        each = exp_config.ExpRecMotivation()
        each.SetLogDir(f"{LOG_PREFIX}/0625-real-motiv-rec-{suffix}")
        exp_lists.append(each)

        # each = exp_config.ExpRealMotivationPerfEmb()
        # each.SetLogDir(f'{LOG_PREFIX}/0625-real-motiv-{suffix}')
        # exp_lists.append(each)

        each = exp_config.ExpKGScalability()
        each.SetLogDir(f"{LOG_PREFIX}/0625-KG-scale-{suffix}")
        exp_lists.append(each)

        each = exp_config.ExpRecPerf()
        each.SetLogDir(f"{LOG_PREFIX}/0625-Rec-{suffix}")
        exp_lists.append(each)

        each = exp_config.ExpMicroPerfEmb()
        each.SetLogDir(f"{LOG_PREFIX}/0625-micro-emb-{suffix}")
        exp_lists.append(each)

        # each = exp_config.ExpRecPerfvsA30()
        # each.SetLogDir(f'{LOG_PREFIX}/0625-RecvsA30-{suffix}')
        # each.SetFilter(lambda config: config['emb_choice'] != 'KnownLocalCachedEmbedding')
        # exp_lists.append(each)

        # each = exp_config.ExpKGvsA30()
        # each.SetLogDir(f'{LOG_PREFIX}/0625-KGvsA30-{suffix}')
        # each.SetFilter(lambda config: config['cached_emb_type'] != 'KnownLocalCachedEmbedding')
        # exp_lists.append(each)

    else:  # 3090
        # 大数据集KG
        each = exp_config.ExpKGSensitiveModel()
        each.SetLogDir(f"{LOG_PREFIX}/1022-kgsen-model-{suffix}")
        # each.SetFilter(lambda config: config['nr_gpus']<=6)
        exp_lists.append(each)


        # sensitive
        each = exp_config.ExpRecSensitiveModelPerf()
        each.SetLogDir(f"{LOG_PREFIX}/1022-rec-sen-model-{suffix}")
        exp_lists.append(each)

        each = exp_config.ExpRecSensitiveFlushThreads()
        each.SetLogDir(f"{LOG_PREFIX}/1022-rec-sen-threads-{suffix}")
        exp_lists.append(each)


		# 大数据集REC
        each = exp_config.ExpRecPerf()
        each.SetLogDir(f"{LOG_PREFIX}/1022-Rec-large-{suffix}")
        each.SetFilter(lambda config: config["dataset"] == "criteoTB")
        exp_lists.append(each)

        # # 用这个
        each = exp_config.ExpMicroPerfEmb()
        each.SetLogDir(f"{LOG_PREFIX}/1022-micro-{suffix}")
        exp_lists.append(each)

        each = exp_config.ExpRecPerf()
        each.SetLogDir(f"{LOG_PREFIX}/1022-Rec-{suffix}")
        each.SetFilter(lambda config: config["dataset"] != "criteoTB")
        exp_lists.append(each)

        each = exp_config.ExpKGScalability()
        each.SetLogDir(f"{LOG_PREFIX}/1022-KG-scale-large-{suffix}")
        each.SetFilter(lambda config: config["dataset"] == "wikikg90M")
        # each.SetFilter(lambda config: config['nr_gpus']<=6)
        exp_lists.append(each)

        each = exp_config.ExpKGSensitiveFlushThreads()
        each.SetLogDir(f"{LOG_PREFIX}/1022-kgsen-threads-{suffix}")
        # each.SetFilter(lambda config: config['nr_gpus']<=6)
        exp_lists.append(each)

        return exp_lists

        # each = exp_config.ExpRealMotivationPerfEmb()
        # each.SetLogDir(f'{LOG_PREFIX}/1011-real-motiv-{suffix}')
        # exp_lists.append(each)

        each = exp_config.ExpRecMotivation()
        each.SetLogDir(f"{LOG_PREFIX}/1011-real-motiv-rec-{suffix}")
        exp_lists.append(each)

        each = exp_config.ExpRecPerfvsA30()
        each.SetLogDir(f"{LOG_PREFIX}/1011-RecVSA30-{suffix}")
        # each.SetFilter(lambda config: config['emb_choice'] == 'KnownLocalCachedEmbedding')
        exp_lists.append(each)

        each = exp_config.ExpKGScalability()
        each.SetLogDir(f"{LOG_PREFIX}/1003-KG-scale-{suffix}")
        each.SetFilter(lambda config: config["dataset"] == "FB15k")
        exp_lists.append(each)

        each = exp_config.ExpKGScalability()
        each.SetLogDir(f"{LOG_PREFIX}/1003-KG-scale-Freebase-{suffix}")
        each.SetFilter(lambda config: config["dataset"] == "Freebase")
        exp_lists.append(each)

        each = exp_config.ExpKGvsA30()
        each.SetLogDir(f"{LOG_PREFIX}/1003-KGvsA30-{suffix}")
        # each.SetFilter(lambda config: config['cached_emb_type'] == 'KnownLocalCachedEmbedding')
        exp_lists.append(each)

        # each = exp_config.ExpRecPerfDebug()
        # each.SetLogDir(f'{LOG_PREFIX}/0510-debugrec-{suffix}')

        # each = exp_config.ExpMicroDebug()
        # each.SetLogDir(f'{LOG_PREFIX}/0510-debugmicro-{suffix}')
        # exp_lists.append(each)
        # return exp_lists

        # each = exp_config.ExpKGScalabilityDecoupled()
        # each.SetLogDir(f'{LOG_PREFIX}/0510-KG-scale-decoupled-{suffix}')
        # exp_lists.append(each)

        # each = exp_config.ExpKGPerfDebug()
        # each.SetLogDir(f'{LOG_PREFIX}/0128-KG-debugomp{suffix}')
        # exp_lists.append(each)

        # # 别用下面的了
        # # each = exp_config.ExpMacroPerfEmb()
        # # each.SetLogDir(f'{LOG_PREFIX}/0117-exp1-macro-perf-emb-{suffix}')
        # # exp_lists.append(each)

    return exp_lists


if __name__ == "__main__":
    exp_lists = main()
    for i, each in enumerate(exp_lists):
        # mount NFS
        mount_master(
            [each for each in ALL_SERVERS_INCLUDING_NOT_USED if each != "127.0.0.1"]
        )
        config_each_server(
            [each for each in ALL_SERVERS_INCLUDING_NOT_USED if each != "127.0.0.1"]
        )

        print("=================-====================")
        print(
            f"Experiment {i}/{len(exp_lists)}: ",
            each.name,
        )

        each.RunExperiment()
