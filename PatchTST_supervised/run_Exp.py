import argparse
import os
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np

np.Inf = np.inf


def main(custom_args=None):
    """
    스크립트의 모든 실행 로직을 포함하는 메인 함수입니다.
    custom_args가 제공되면 이를 사용하여 argparse를 오버라이드할 수 있습니다.
    """
    parser = argparse.ArgumentParser(
        description="Autoformer & Transformer family for Time Series Forecasting"
    )

    # --- Argument Parsers ---

    # random seed
    parser.add_argument("--random_seed", type=int, default=2021, help="random seed")

    # basic config
    parser.add_argument(
        "--is_training", type=int, default=1, help="status (1: train, 0: test)"
    )
    parser.add_argument("--model_id", type=str, default="test", help="model id")
    parser.add_argument(
        "--model",
        type=str,
        default="Autoformer",
        help="model name, options: [Autoformer, Informer, Transformer, PatchTST]",
    )

    # data loader
    parser.add_argument("--data", type=str, default="ETTm1", help="dataset type")
    parser.add_argument(
        "--root_path",
        type=str,
        default="./data/ETT/",
        help="root path of the data file",
    )
    parser.add_argument("--data_path", type=str, default="ETTh1.csv", help="data file")
    parser.add_argument(
        "--features", type=str, default="M", help="forecasting task, options:[M, S, MS]"
    )
    parser.add_argument(
        "--target", type=str, default="OT", help="target feature in S or MS task"
    )
    parser.add_argument(
        "--freq", type=str, default="h", help="freq for time features encoding"
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        default="./checkpoints/",
        help="location of model checkpoints",
    )

    # forecasting task
    parser.add_argument("--seq_len", type=int, default=96, help="input sequence length")
    parser.add_argument("--label_len", type=int, default=48, help="start token length")
    parser.add_argument(
        "--pred_len", type=int, default=96, help="prediction sequence length"
    )

    # PatchTST specific
    parser.add_argument(
        "--fc_dropout", type=float, default=0.05, help="fully connected dropout"
    )
    parser.add_argument("--head_dropout", type=float, default=0.0, help="head dropout")
    parser.add_argument("--patch_len", type=int, default=16, help="patch length")
    parser.add_argument("--stride", type=int, default=8, help="stride")
    parser.add_argument(
        "--padding_patch", default="end", help="None: None; end: padding on the end"
    )
    parser.add_argument("--revin", type=int, default=1, help="RevIN; True 1 False 0")
    parser.add_argument(
        "--affine", type=int, default=0, help="RevIN-affine; True 1 False 0"
    )
    parser.add_argument(
        "--subtract_last",
        type=int,
        default=0,
        help="0: subtract mean; 1: subtract last",
    )
    parser.add_argument(
        "--decomposition", type=int, default=0, help="decomposition; True 1 False 0"
    )
    parser.add_argument(
        "--kernel_size", type=int, default=25, help="decomposition-kernel"
    )
    parser.add_argument(
        "--individual", type=int, default=0, help="individual head; True 1 False 0"
    )

    # Formers (Common Architectures)
    parser.add_argument("--embed_type", type=int, default=0, help="embedding type")
    parser.add_argument("--enc_in", type=int, default=7, help="encoder input size")
    parser.add_argument("--dec_in", type=int, default=7, help="decoder input size")
    parser.add_argument("--c_out", type=int, default=7, help="output size")
    parser.add_argument("--d_model", type=int, default=512, help="dimension of model")
    parser.add_argument("--n_heads", type=int, default=8, help="num of heads")
    parser.add_argument("--e_layers", type=int, default=2, help="num of encoder layers")
    parser.add_argument("--d_layers", type=int, default=1, help="num of decoder layers")
    parser.add_argument("--d_ff", type=int, default=2048, help="dimension of fcn")
    parser.add_argument(
        "--moving_avg", type=int, default=25, help="window size of moving average"
    )
    parser.add_argument("--factor", type=int, default=1, help="attn factor")
    parser.add_argument(
        "--distil",
        action="store_false",
        default=True,
        help="whether to use distilling in encoder",
    )
    parser.add_argument("--dropout", type=float, default=0.05, help="dropout")
    parser.add_argument(
        "--embed", type=str, default="timeF", help="time features encoding"
    )
    parser.add_argument("--activation", type=str, default="gelu", help="activation")
    parser.add_argument(
        "--output_attention",
        action="store_true",
        help="whether to output attention in ecoder",
    )
    parser.add_argument(
        "--do_predict",
        action="store_true",
        help="whether to predict unseen future data",
    )

    # optimization
    parser.add_argument(
        "--num_workers", type=int, default=10, help="data loader num workers"
    )
    parser.add_argument("--itr", type=int, default=2, help="experiments times")
    parser.add_argument("--train_epochs", type=int, default=100, help="train epochs")
    parser.add_argument(
        "--batch_size", type=int, default=128, help="batch size of train input data"
    )
    parser.add_argument(
        "--patience", type=int, default=100, help="early stopping patience"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.0001, help="optimizer learning rate"
    )
    parser.add_argument("--des", type=str, default="test", help="exp description")
    parser.add_argument("--loss", type=str, default="mse", help="loss function")
    parser.add_argument(
        "--lradj", type=str, default="type3", help="adjust learning rate"
    )
    parser.add_argument("--pct_start", type=float, default=0.3, help="pct_start")
    parser.add_argument(
        "--use_amp",
        action="store_true",
        help="use automatic mixed precision training",
        default=False,
    )

    # GPU
    parser.add_argument("--use_gpu", type=bool, default=True, help="use gpu")
    parser.add_argument("--gpu", type=int, default=0, help="gpu")
    parser.add_argument(
        "--use_multi_gpu", action="store_true", help="use multiple gpus", default=False
    )
    parser.add_argument(
        "--devices", type=str, default="0,1,2,3", help="device ids of multile gpus"
    )
    parser.add_argument(
        "--test_flop",
        action="store_true",
        default=False,
        help="See utils/tools for usage",
    )

    # args 처리

    args = parser.parse_args(custom_args)

    # --- Setup and Initialization ---

    # 1. Random Seed Fixing
    fix_seed = args.random_seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    # 2. GPU Setup
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        # 'devices' 오타 수정: args.dvices -> args.devices
        args.devices = args.devices.replace(" ", "")
        device_ids = args.devices.split(",")
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print("Args in experiment:")
    print(args)

    Exp = Exp_Main

    # --- Training and Testing Logic ---

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            setting = "{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}".format(
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.des,
                ii,
            )

            exp = Exp(args)  # set experiments
            print(
                ">>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>".format(setting)
            )
            exp.train(setting)

            print(
                ">>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(setting)
            )
            exp.test(setting)

            if args.do_predict:
                print(
                    ">>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(
                        setting
                    )
                )
                exp.predict(setting, True)

            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = "{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}".format(
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.distil,
            args.des,
            ii,
        )

        exp = Exp(args)  # set experiments
        print(">>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()


# --- Main Execution Block ---
if __name__ == "__main__":
    main()
