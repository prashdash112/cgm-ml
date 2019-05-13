'''
This script trains PointNet.
'''
from cgmcore import modelutils
from cgmcore import utils
import numpy as np
from keras import callbacks, optimizers
import pprint
import os
from cgmcore.preprocesseddatagenerator import get_dataset_path, create_datagenerator_from_parameters
import random
import qrcodes

# Get the dataset path.
dataset_path = get_dataset_path()
print("Using dataset path", dataset_path)

# Hyperparameters.
steps_per_epoch = 100
validation_steps = 10
epochs = 100
batch_size = 8
random_seed = 667

# For creating pointclouds.
dataset_parameters_pointclouds = {}
dataset_parameters_pointclouds["input_type"] = "pointcloud"
dataset_parameters_pointclouds["output_targets"] = ["height"]
dataset_parameters_pointclouds["random_seed"] = random_seed
dataset_parameters_pointclouds["pointcloud_target_size"] = 10000
dataset_parameters_pointclouds["pointcloud_random_rotation"] = False
dataset_parameters_pointclouds["sequence_length"] = 0
datagenerator_instance_pointclouds = create_datagenerator_from_parameters(dataset_path, dataset_parameters_pointclouds)

# Get the QR-codes.
#qrcodes_to_use = datagenerator_instance_pointclouds.qrcodes[:]
qrcodes_to_use = qrcodes.standing_list

# Do the split.
random.seed(random_seed)
qrcodes_shuffle = qrcodes_to_use[:]
random.shuffle(qrcodes_shuffle)
split_index = int(0.8 * len(qrcodes_shuffle))
qrcodes_train = sorted(qrcodes_shuffle[:split_index])
qrcodes_validate = sorted(qrcodes_shuffle[split_index:])
del qrcodes_shuffle
print("QR-codes for training:\n", "\t".join(qrcodes_train))
print("QR-codes for validation:\n", "\t".join(qrcodes_validate))

# Create python generators.
generator_pointclouds_train = datagenerator_instance_pointclouds.generate(size=batch_size, qrcodes_to_use=qrcodes_train)
generator_pointclouds_validate = datagenerator_instance_pointclouds.generate(size=batch_size, qrcodes_to_use=qrcodes_validate)

# Testing the genrators.
def test_generator(generator):
    data = next(generator)
    print("Input:", data[0].shape, "Output:", data[1].shape)
test_generator(generator_pointclouds_train)
test_generator(generator_pointclouds_validate)

# Training details.
training_details = {
    "dataset_path" : dataset_path,
    "qrcodes_train" : qrcodes_train,
    "qrcodes_validate" : qrcodes_validate,
    "steps_per_epoch" : steps_per_epoch,
    "validation_steps" : validation_steps,
    "epochs" : epochs,
    "batch_size" : batch_size,
    "random_seed" : random_seed,
}

# Output path. Ensure its existence.
output_path = "models"
if os.path.exists(output_path) == False:
    os.makedirs(output_path)
print("Using output path:", output_path)

# Important things.
pp = pprint.PrettyPrinter(indent=4)
tensorboard_callback = callbacks.TensorBoard()
histories = {}
    
# Training PointNet.
def train_pointnet():

    input_shape = (dataset_parameters_pointclouds["pointcloud_target_size"], 3)
    output_size = 1
    model_pointnet = modelutils.create_point_net(input_shape, output_size, hidden_sizes = [512, 256, 128])
    model_pointnet.summary()
    
    # Compile the model.
    #optimizer = "rmsprop"
    optimizer = optimizers.Adagrad(lr=0.7, epsilon=None, decay=0.2)
    model_pointnet.compile(
            optimizer=optimizer,
            loss="mse",
            metrics=["mae"]
        )

    # Train the model.
    history = model_pointnet.fit_generator(
        generator_pointclouds_train,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=generator_pointclouds_validate,
        validation_steps=validation_steps,
        callbacks=[tensorboard_callback]
        )

    histories["pointnet"] = history
    modelutils.save_model_and_history(output_path, model_pointnet, history, training_details, "pointnet")

train_pointnet()