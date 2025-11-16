import argparse
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ExpConfig:
    """Configuration class mirroring arguments from the main experiment script."""

    # --- Random Seed ---
    random_seed: int = 2021

    # --- Basic Config ---
    is_training: int = 1
    model_id: str = "test"
    model: str = "Autoformer"

    # --- Data Loader ---
    data: str = "ETTm1"
    root_path: str = "./data/ETT/"
    data_path: str = "ETTh1.csv"
    features: str = "M"
    target: str = "OT"
    freq: str = "h"
    checkpoints: str = "./checkpoints/"

    # --- Forecasting Task ---
    seq_len: int = 96
    label_len: int = 48
    pred_len: int = 96

    # --- PatchTST ---
    fc_dropout: float = 0.05
    head_dropout: float = 0.0
    patch_len: int = 16
    stride: int = 8
    padding_patch: str = "end"
    revin: int = 1
    affine: int = 0
    subtract_last: int = 0
    decomposition: int = 0
    kernel_size: int = 25
    individual: int = 0

    # --- Formers ---
    embed_type: int = 0
    enc_in: int = 7
    dec_in: int = 7
    c_out: int = 7
    d_model: int = 512
    n_heads: int = 8
    e_layers: int = 2
    d_layers: int = 1
    d_ff: int = 2048
    moving_avg: int = 25
    factor: int = 1

    distil: bool = True

    dropout: float = 0.05
    embed: str = "timeF"
    activation: str = "gelu"

    output_attention: bool = False
    do_predict: bool = False

    # --- Optimization ---
    num_workers: int = 10
    itr: int = 2
    train_epochs: int = 100
    batch_size: int = 128
    patience: int = 100
    learning_rate: float = 0.0001
    des: str = "test"
    loss: str = "mse"
    lradj: str = "type3"
    pct_start: float = 0.3

    use_amp: bool = False

    # --- GPU ---
    use_gpu: bool = False
    gpu: int = 0

    use_multi_gpu: bool = False
    devices: str = "0,1,2,3"

    test_flop: bool = False

    # Reserved slots for other dynamic attributes (e.g., set in exp_main)
    device_ids: list[int] = field(default_factory=list)
    device: Any = None

    @classmethod
    def from_args(cls, args: argparse.Namespace):
        args = vars(args)
        return cls(**args)

    def to_args(self):
        data_dict = asdict(self)
        args = argparse.Namespace(**data_dict)
        return args


def generate_args(custom_args: list[str] | dict[str, Any] | None = None):
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
    parser.add_argument("--use_gpu", type=bool, default=False, help="use gpu")
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

    # args processing
    if isinstance(custom_args, dict):
        args = parser.parse_args()
        for k, v in custom_args.items():
            setattr(args, k, v)
    else:
        args = parser.parse_args(custom_args)

    if args.use_gpu and args.use_multi_gpu:
        # device typo correction: args.devices is used
        args.devices = args.devices.replace(" ", "")
        device_ids = args.devices.split(",")
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    return args


# --- Usage Example: Creating Weather Args ---


def create_weather_config(pred_len: int = 96) -> ExpConfig:
    """
    Creates an ExpConfig object with core settings for PatchTST/Weather dataset.
    """
    config = ExpConfig()

    config.model = "PatchTST"
    config.data = "custom"
    config.root_path = "./dataset/"
    config.data_path = "weather.csv"
    config.features = "M"
    config.seq_len = 336
    config.label_len = 0
    config.pred_len = pred_len

    # PatchTST/Weather architecture overrides
    config.enc_in = 21
    config.e_layers = 3
    config.n_heads = 16
    config.d_model = 128
    config.d_ff = 256
    config.dropout = 0.2
    config.fc_dropout = 0.2
    config.head_dropout = 0.0
    config.patch_len = 16
    config.stride = 8

    # Training overrides
    config.train_epochs = 100
    config.patience = 20
    config.batch_size = 128
    config.learning_rate = 0.0001

    # Dynamic model_id setting
    config.model_id = f"weather_{config.seq_len}_{pred_len}"

    return config
