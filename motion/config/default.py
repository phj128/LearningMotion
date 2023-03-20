import yacs.config


class Config(yacs.config.CfgNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, new_allowed=True)


CN = Config


_C = CN()

_C.TYPE = "DefaultTrainer"

STATE_DIM = 647
ACTION_NUMBER = 5

DATASET = CN()
DATASET.TYPE = "SAMPData"
dataset_cfg = CN()
dataset_cfg.data_dir = "./datasets/samp/MotionNet"
dataset_cfg.state_dim = STATE_DIM
dataset_cfg.L = 60
dataset_cfg.is_scheduled_sampling = False

dataset_cfg.collate_fn = "default_collate"
dataset_cfg.valid_collate_fn = "default_collate"
dataset_cfg.valid_batch_size = 1

DATASET.cfg = dataset_cfg


DATALOADER = CN()
DATALOADER.batch_size = 32
DATALOADER.num_workers = 2

MODEL = CN()
MODEL.TYPE = "MotionPredictor"

MODEL.cfg = CN()
MODEL.cfg.ENCODER_TYPE = "SAMPEncoder"

MODEL.cfg.state_dim = STATE_DIM
MODEL.cfg.dropout = 0.0
MODEL.cfg.I_in_dim = 2048
MODEL.cfg.I_out_dim = 256
MODEL.cfg.h_dim = 512
MODEL.cfg.pred_in_dim = 903
MODEL.cfg.activation = "ELU"

MODEL.cfg.MoE_layernum = 3
MODEL.cfg.num_experts = 12
MODEL.cfg.h_dim_gate = 512

MODEL.cfg.z_dim = 64
MODEL.cfg.act = "Exp"

PIPELINE = CN()
PIPELINE.TYPE = "RegressionPipeline"
PIPELINE.cfg = CN()
PIPELINE.cfg.num_actions = ACTION_NUMBER

PIPELINE.cfg.func_wrap_pairs = [
    ["run_step", "sequence_process"]
]  # sequnce w/o gradients accumulation
PIPELINE.cfg.func_wrap_pairs = []


LOSS = CN()
RECONSTRUCTION = CN()
RECONSTRUCTION.TYPE = "mseloss"
RECONSTRUCTION.weight = 1.0
RECONSTRUCTION.reduction = "sum"
RECONSTRUCTION.average_dim = [0]
LOSS.RECONSTRUCTION = RECONSTRUCTION

KLD = CN()
KLD.TYPE = "kldloss"
KLD.weight = 0.1
KLD.reduction = "sum"
KLD.average_dim = [0]
LOSS.KLD = KLD

PIPELINE.cfg.LOSS = LOSS


CRITERION = CN()
CRITERION.TYPE = "RegressionCriterion"
CRITERION.weight = 1.0
CRITERION.cfg = CN()


OPTIMIZER = CN()
OPTIMIZER.TYPE = "Adam"
OPTIMIZER.lr = 5e-5  # SAMP
# OPTIMIZER.betas = (0.9, 0.999)

SCHEDULE = CN()
SCHEDULE.max_epochs = 100
SCHEDULE.lr = 1e-4
SCHEDULE.min_lr = 5e-6
scheduler = CN()
scheduler.TYPE = "LambdaLR"
scheduler.lr_lambda = "linear_epoch_func"
SCHEDULE.scheduler = scheduler


HOOK = CN()
hook_save = CN()
hook_save.TYPE = "SaveCkptHook"
hook_save.priority = 10
hook_save.interval = 100
HOOK.hook_save = hook_save
hook_ss = CN()
hook_ss.TYPE = "ScheduledSamplingHook"
hook_ss.priority = 20
hook_ss.milestones = [30, 60]
# HOOK.hook_ss = hook_ss
hook_anneal = CN()
hook_anneal.TYPE = "AnnealingHook"
hook_anneal.priority = 30
hook_anneal.annealing_pairs = []
# hook_anneal.annealing_pairs = [["KLD", [0, 25]]]
HOOK.hook_anneal = hook_anneal
hook_clip_grad = CN()
hook_clip_grad.TYPE = "ClipGradHook"
hook_clip_grad.priority = 10
hook_clip_grad.max_grad = None
HOOK.hook_clip_grad = hook_clip_grad
hook_eval = CN()
hook_eval.TYPE = "EvalHook"
hook_eval.priority = 80
hook_eval.interval = 1000
hook_eval.test_before_train = False
HOOK.hook_eval = hook_eval
hook_log = CN()
hook_log.TYPE = "LogShowHook"
hook_log.priority = 90
HOOK.hook_log = hook_log
hook_tb = CN()
hook_tb.TYPE = "TensorBoardHook"
hook_tb.priority = 100
HOOK.hook_tb = hook_tb

ONNX = CN()
ONNX.input_names = ["Z", "Cond"]
ONNX.input_shapes = [[64], [2048]]
ONNX.output_names = ["Y_hat"]

cfg = CN()
cfg.output_dir = "work_dirs/debug"

cfg.DATASET = DATASET
cfg.DATALOADER = DATALOADER
cfg.MODEL = MODEL
cfg.PIPELINE = PIPELINE
cfg.CRITERION = CRITERION
cfg.OPTIMIZER = OPTIMIZER
cfg.SCHEDULE = SCHEDULE
cfg.HOOK = HOOK
cfg.ONNX = ONNX

_C.cfg = cfg
