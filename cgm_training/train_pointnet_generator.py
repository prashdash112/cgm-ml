'''
This script trains PointNet.
'''
import sys
sys.path.insert(0, "..")




import warnings
warnings.filterwarnings("ignore")
from cgmcore import modelutils
from cgmcore import utils
import numpy as np
from tensorflow.keras import callbacks, optimizers, models, layers
import tensorflow as tf
import pprint
import os
from cgmcore.preprocesseddatagenerator import get_dataset_path, create_datagenerator_from_parameters
import random
import qrcodes
from cgmcore.utils import create_training_tasks
import multiprocessing

tf.get_logger().setLevel('WARNING')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}


# parse the arguments
import argparse
parser = argparse.ArgumentParser(description='Training on gpu')
parser.add_argument('-dataset_path',      action="store",      dest="dataset_path",      type=str, help='path to dataset')
parser.add_argument('-model_path',        action="store",      dest="model_path",        type=str, help='set path to model for retraining')
parser.add_argument('-training_target',   action="store",      dest="training_target",   type=str, help='select WEIGHT or HEIGHT')
parser.add_argument('-use_multi_gpu',     action="store_true", dest="use_multi_gpu",               help='set the training on multiple gpus')

# unused till now
parser.add_argument('-epochs',            action="store",      dest="epochs",           default=2000, type=int, help='nr. of epochs for training')
parser.add_argument('-batch_size',        action="store",      dest="batch_size",       default=16,   type=int, help='batch size of training')
parser.add_argument('-steps_per_epoch',   action="store",      dest="steps_per_epoch",  default=50,   type=int, help='nr. of steps per epoche')
parser.add_argument('-validation_steps',  action="store",      dest="validation_steps", default=20,   type=int, help='steps for validation')




config = parser.parse_args()
print ('dataset_path       =', config.dataset_path)
print ('model_path         =', config.model_path)
print ('use_multi_gpu      =', config.use_multi_gpu)
print ('training_target    =', config.training_target)

# training parameter
print ('epochs             =', config.epochs)
print ('batch_size         =', config.batch_size)
print ('steps_per_epoch    =', config.steps_per_epoch)
print ('validation_steps   =', config.validation_steps)




# Get the dataset path.
# if len(config.dataset_path) == 1:
#     dataset_path = get_dataset_path()
# else:
dataset_path = config.dataset_path
print("Using dataset path", dataset_path)

output_root_path = "/whhdata/models"

# Hyperparameters.
steps_per_epoch = 50
validation_steps = 20
epochs = 2000
batch_size = 16
random_seed = 667
use_early_stoping = False
early_stopping_threshold = 0.0001 


if len(utils.get_available_gpus()) == 0:
    output_root_path = "."
    steps_per_epoch = 1
    validation_steps = 1
    epochs = 2
    batch_size = 1
    random_seed = 667
    print("WARNING! No GPU available!")

    
# For creating pointclouds.
dataset_parameters = {}
dataset_parameters["input_type"] = "fusion"
dataset_parameters["output_targets"] = ["weight"]
dataset_parameters["random_seed"] = random_seed
dataset_parameters["pointcloud_target_size"] = 10000
dataset_parameters["pointcloud_random_rotation"] = False
dataset_parameters["sequence_length"] = 0
dataset_parameters["pointcloud_subsampling_method"] = "random"

datagenerator_instance = create_datagenerator_from_parameters(dataset_path, dataset_parameters)







# Get the QR-codes.
qrcodes = datagenerator_instance.qrcodes[:]
subset_sizes = [1.0]    
qrcodes_tasks = create_training_tasks(qrcodes, subset_sizes)

# Go through all.
for qrcodes_task in qrcodes_tasks:
    
    qrcodes_train, qrcodes_validate = qrcodes_task
    print("Using {} QR-codes for training.".format(len(qrcodes_train))) 
    print("Using {} QR-codes for validation.".format(len(qrcodes_validate)))

    # Create python generators.
    workers = 4
    generator_train = datagenerator_instance.generate(
        size=batch_size, 
        qrcodes_to_use=qrcodes_train, 
        workers=workers
    )
    generator_validate = datagenerator_instance.generate(
        size=batch_size,
        qrcodes_to_use=qrcodes_validate,
        workers=workers)

    # Testing the genrators.
    def test_generator(generator):
        data = next(generator)
        print("Input:", data[0].shape, "Output:", data[1].shape)
    test_generator(generator_train)
    test_generator(generator_validate)

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
        "dataset_parameters" : dataset_parameters
    }

    # Date time string.
    datetime_string = utils.get_datetime_string() + "_{}-{}".format(len(qrcodes_train), len(qrcodes_validate)) + "_".join(dataset_parameters["output_targets"])

    # Output path. Ensure its existence.
    output_path = os.path.join(output_root_path, datetime_string)
    if os.path.exists(output_path) == False:
        os.makedirs(output_path)
    print("Using output path:", output_path)

    # Important things.
    pp = pprint.PrettyPrinter(indent=4)
    log_dir = os.path.join("/whhdata/models", "logs", datetime_string)
#    tensorboard_callback = callbacks.TensorBoard(log_dir=log_dir)
    histories = {}

    # Training network.
    def train_pointclouds():
        data_size = 3
        if(dataset_parameters["input_type"] == "fusion"):
            data_size = 7

        input_shape = (dataset_parameters["pointcloud_target_size"], data_size)
        output_size = 1
        if(config.model_path): 
            model = tf.keras.models.load_model(config.model_path)
            print('Continue to train on loaded model from ' + str(config.model_path))
        else:
            model = modelutils.create_point_net(input_shape, output_size, hidden_sizes = [512, 256, 128, 64])
            print('Creating new model')

        model.summary()

        # Compile the model.
        #optimizer = optimizers.RMSprop(lr=0.0001)
        optimizer = optimizers.RMSprop(learning_rate=0.0001)
        # optimizer = optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)

        if(config.use_multi_gpu):
            model = tf.keras.utils.multi_gpu_model(model, gpus=2)
        
        model.compile(
                optimizer=optimizer,
                loss="mse",
                metrics=["mae"]
            )

        try:
            tensorboard_callback = tf.keras.callbacks.TensorBoard(
                                log_dir=log_dir, 
                                histogram_freq=0, 
                                batch_size=32, 
                                write_graph=True, 
                                write_grads=False, 
                                write_images=True, 
                                embeddings_freq=0, 
                                embeddings_layer_names=None, 
                                embeddings_metadata=None, 
                                embeddings_data=None, 
                                update_freq='epoch')


            early_stopping = tf.keras.callbacks.EarlyStopping(
                                monitor='val_loss',
                                min_delta=early_stopping_threshold,
                                patience=5,
                                verbose=1)

            val_loss_callback = tf.keras.callbacks.ModelCheckpoint(output_path + "/" + datetime_string + "best_weights.{epoch:5d}-{val_loss:.2f}.hdf5", 
                                monitor='val_loss', 
                                verbose=0, 
                                save_best_only=True, 
                                save_weights_only=False, 
                                mode='auto', 
                                save_freq='epoch')

            # Train the model.
            history = model.fit_generator(
                generator_train,
                steps_per_epoch=steps_per_epoch,
                epochs=epochs,
                validation_data=generator_validate,
                validation_steps=validation_steps,
                use_multiprocessing=False,
                workers=0,
                callbacks=[tensorboard_callback, val_loss_callback]
                )
        except KeyboardInterrupt:
            print("ALAAAARM")
            histories["pointnet"] = history
            modelutils.save_model_and_history(output_path, datetime_string, model, history, training_details, "pointnet")
            datagenerator_instance.finish()
            

        histories["pointnet"] = history
        modelutils.save_model_and_history(output_path, datetime_string, model, history, training_details, "pointnet")

    train_pointclouds()
