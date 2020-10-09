import os
import random
from typing import List

import glob2 as glob
import tensorflow as tf
from azureml.core import Experiment, Workspace
from azureml.core.run import Run
from tensorflow.keras import callbacks, layers, models

from config import CONFIG
from constants import REPO_DIR
from model import create_base_cnn, create_head, load_base_cgm_model
from preprocessing import create_multiartifact_paths, tf_load_pickle, tf_augment_sample


# Make experiment reproducible
tf.random.set_seed(CONFIG.SPLIT_SEED)
random.seed(CONFIG.SPLIT_SEED)

# Get the current run.
run = Run.get_context()

# Offline run. Download the sample dataset and run locally. Still push results to Azure.
if(run.id.startswith("OfflineRun")):
    print("Running in offline mode...")

    # Access workspace.
    print("Accessing workspace...")
    workspace = Workspace.from_config()
    experiment = Experiment(workspace, "training-junkyard")
    run = experiment.start_logging(outputs=None, snapshot_directory=None)

    # Get dataset.
    print("Accessing dataset...")
    dataset_name = "anon-depthmap-mini"
    dataset_path = str(REPO_DIR / "data" / dataset_name)
    if not os.path.exists(dataset_path):
        dataset = workspace.datasets[dataset_name]
        dataset.download(target_path=dataset_path, overwrite=False)

# Online run. Use dataset provided by training notebook.
else:
    print("Running in online mode...")
    experiment = run.experiment
    workspace = experiment.workspace
    dataset_path = run.input_datasets["dataset"]

# Get the QR-code paths.
dataset_scans_path = os.path.join(dataset_path, "scans")
print("Dataset path:", dataset_scans_path)
# print(glob.glob(os.path.join(dataset_scans_path, "*"))) # Debug
print("Getting QR-code paths...")
qrcode_paths = glob.glob(os.path.join(dataset_scans_path, "*"))
print("qrcode_paths: ", len(qrcode_paths))
assert len(qrcode_paths) != 0

# Shuffle and split into train and validate.
random.seed(CONFIG.SPLIT_SEED)
random.shuffle(qrcode_paths)
split_index = int(len(qrcode_paths) * 0.8)
qrcode_paths_training = qrcode_paths[:split_index]

qrcode_paths_validate = qrcode_paths[split_index:]
qrcode_paths_activation = random.choice(qrcode_paths_validate)
qrcode_paths_activation = [qrcode_paths_activation]

del qrcode_paths

# Show split.
print("Paths for training:")
print("\t" + "\n\t".join(qrcode_paths_training))
print("Paths for validation:")
print("\t" + "\n\t".join(qrcode_paths_validate))
print("Paths for activation:")
print("\t" + "\n\t".join(qrcode_paths_activation))

print(len(qrcode_paths_training))
print(len(qrcode_paths_validate))

assert len(qrcode_paths_training) > 0 and len(qrcode_paths_validate) > 0


def create_samples(qrcode_paths: List[str]) -> List[List[str]]:
    samples = []
    for qrcode_path_training in sorted(qrcode_paths):
        for code in CONFIG.CODES_FOR_POSE_AND_SCANSTEP:
            p = os.path.join(qrcode_path_training, code)
            new_samples = create_multiartifact_paths(p, CONFIG.N_ARTIFACTS)
            samples.extend(new_samples)
    return samples


paths_training = create_samples(qrcode_paths_training)
print(f"Samples for training: {len(paths_training)}")

paths_validate = create_samples(qrcode_paths_validate)
print(f"Samples for validate: {len(paths_validate)}")

paths_activate = create_samples(qrcode_paths_activation)
print(f"Samples for activate: {len(paths_activate)}")

# Create dataset for training.
paths = paths_training  # list
dataset = tf.data.Dataset.from_tensor_slices(paths)  # TensorSliceDataset  # List[ndarray[str]]
dataset = dataset.cache()
dataset = dataset.repeat(CONFIG.N_REPEAT_DATASET)
dataset = dataset.map(
    lambda path: tf_load_pickle(paths=path),
    tf.data.experimental.AUTOTUNE
)  # (240,180,5), (1,)

dataset = dataset.map(tf_augment_sample, tf.data.experimental.AUTOTUNE)

dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
dataset = dataset.shuffle(CONFIG.SHUFFLE_BUFFER_SIZE)
dataset_training = dataset

# Create dataset for validation.
# Note: No shuffle necessary.
paths = paths_validate
dataset = tf.data.Dataset.from_tensor_slices(paths)
dataset_norm = dataset.map(lambda path: tf_load_pickle(path), tf.data.experimental.AUTOTUNE)
dataset_norm = dataset_norm.cache()
dataset_norm = dataset_norm.prefetch(tf.data.experimental.AUTOTUNE)
dataset_validation = dataset_norm
del dataset_norm

# Create dataset for activation
# paths = paths_activate
# dataset = tf.data.Dataset.from_tensor_slices(paths)
# dataset_norm = dataset.map(lambda path: tf_load_pickle(path), tf.data.experimental.AUTOTUNE)
# dataset_norm = dataset_norm.cache()
# dataset_norm = dataset_norm.prefetch(tf.data.experimental.AUTOTUNE)
# dataset_activation = dataset_norm
# del dataset_norm

# Note: Now the datasets are prepared.


def download_pretrained_model(output_model_fpath):
    print(f"Downloading pretrained model from {CONFIG.PRETRAINED_RUN}")
    previous_experiment = Experiment(workspace=workspace, name=CONFIG.PRETRAINED_EXPERIMENT)
    previous_run = Run(previous_experiment, CONFIG.PRETRAINED_RUN)
    previous_run.download_file("outputs/best_model.h5", output_model_fpath)


def get_base_model():
    if CONFIG.PRETRAINED_RUN:
        model_fpath = REPO_DIR / "data/pretrained/" / CONFIG.PRETRAINED_RUN / "best_model.h5"
        if not os.path.exists(model_fpath):
            download_pretrained_model(model_fpath)
        print(f"Loading pretrained model from {model_fpath}")
        base_model = load_base_cgm_model(model_fpath, should_freeze=CONFIG.SHOULD_FREEZE_BASE)
    else:
        input_shape = (CONFIG.IMAGE_TARGET_HEIGHT, CONFIG.IMAGE_TARGET_WIDTH, 1)
        base_model = create_base_cnn(input_shape, dropout=CONFIG.USE_DROPOUT)  # output_shape: (128,)
    return base_model


# Create the base model
base_model = get_base_model()
base_model.summary()

# Create the head
head_input_shape = (128 * CONFIG.N_ARTIFACTS,)
head_model = create_head(head_input_shape, dropout=CONFIG.USE_CROPOUT)
# head_model.summary()

# Implement artifact flow through the same model
model_input = layers.Input(
    shape=(CONFIG.IMAGE_TARGET_HEIGHT, CONFIG.IMAGE_TARGET_WIDTH, CONFIG.N_ARTIFACTS)
)

features_list = []
for i in range(CONFIG.N_ARTIFACTS):
    features_part = model_input[:, :, :, i:i + 1]
    features_part = base_model(features_part)
    features_list.append(features_part)

concatenation = tf.concat(features_list, axis=-1)
model_output = layers.Dense(1, activation="linear")(concatenation)  # shape: (None,640)

model = models.Model(model_input, model_output)
model.summary()

# Get ready to add callbacks.
training_callbacks = []


# Pushes metrics and losses into the run on AzureML.
class AzureLogCallback(callbacks.Callback):

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            for key, value in logs.items():
                run.log(key, value)


training_callbacks.append(AzureLogCallback())


# Add TensorBoard callback.
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir="logs",
    histogram_freq=0,
    write_graph=True,
    write_grads=False,
    write_images=True,
    embeddings_freq=0,
    embeddings_layer_names=None,
    embeddings_metadata=None,
    embeddings_data=None,
    update_freq="epoch"
)
training_callbacks.append(tensorboard_callback)

# Add checkpoint callback.
#best_model_path = os.path.join('validation','best_model.h5')
best_model_path = str(REPO_DIR / 'data/outputs/best_model.h5')
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=best_model_path,
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)
training_callbacks.append(checkpoint_callback)

layer_name = 'conv2d_11'
#save_dir = os.path.join('validation','out')
save_dir = str(REPO_DIR / 'data/outputs/out')
CHECK_FOLDER = os.path.isdir(save_dir)
#if not CHECK_FOLDER:
#    os.makedirs(save_dir)
#    print("created folder : ", save_dir)

optimizer = tf.keras.optimizers.Nadam(learning_rate=CONFIG.LEARNING_RATE)

# Compile the model.
model.compile(
    optimizer=optimizer,
    loss="mse",
    metrics=["mae"]
)

# Train the model.
model.fit(
    dataset_training.batch(CONFIG.BATCH_SIZE),
    validation_data=dataset_validation.batch(CONFIG.BATCH_SIZE),
    epochs=CONFIG.EPOCHS,
    callbacks=training_callbacks,
    verbose=2
)

# Done.
run.complete()
