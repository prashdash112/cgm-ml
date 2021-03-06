class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


DATA_AUGMENTATION_SAME_PER_CHANNEL = "same_per_channel"
DATA_AUGMENTATION_DIFFERENT_EACH_CHANNEL = "different_each_channel"
DATA_AUGMENTATION_NO = "no"

SAMPLING_STRATEGY_SYSTEMATIC = "systematic"
SAMPLING_STRATEGY_WINDOW = "window"

CONFIG = dotdict(dict(
    SPLIT_SEED=0,
    IMAGE_TARGET_HEIGHT=240,
    IMAGE_TARGET_WIDTH=180,
    EPOCHS=1000,
    BATCH_SIZE=64,
    SHUFFLE_BUFFER_SIZE=2560,
    NORMALIZATION_VALUE=7.5,
    LEARNING_RATE=0.001,

    # Parameters for dataset generation.
    TARGET_INDEXES=[0],  # 0 is height, 1 is weight.
    N_ARTIFACTS=5,
    CODES_FOR_POSE_AND_SCANSTEP=("100", ),
    N_REPEAT_DATASET=1,
    DATA_AUGMENTATION_MODE=DATA_AUGMENTATION_NO,
    SAMPLING_STRATEGY=SAMPLING_STRATEGY_SYSTEMATIC,
    USE_DROPOUT=False,

    PRETRAINED_EXPERIMENT="q3-depthmap-plaincnn-height-95k",
    # PRETRAINED_RUN="q3-depthmap-plaincnn-height-95k_1600451633_cb44f6db",  # Run17 (baseline: min(val_mae)=2.21cm)
    PRETRAINED_RUN="q3-depthmap-plaincnn-height-95k_1597988908_42c4ef33",  # Run3 (baseline: min(val_mae)=1.96cm)

    SHOULD_FREEZE_BASE=True,
))
