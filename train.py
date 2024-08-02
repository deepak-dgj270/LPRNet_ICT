import tensorflow as tf
from tensorflow import keras
import os
import time
import argparse
import utils
import math
from model import LPRNet
import evaluate
import numpy as np

# Custom training loop supporting variable training labels
def train():
    args = parser_args()
    CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    NUM_CLASS = len(CHARS) + 1

    # Initiate the Neural Network
    net = LPRNet(NUM_CLASS)

    # Get the train and validation batch size from argument parser
    batch_size = args["batch_size"]
    val_batch_size = args["val_batch_size"]

    # Initialize the custom data generator
    # Defined in utils.py
    train_gen = utils.DataIterator(img_dir=args["train_dir"], batch_size=batch_size)
    val_gen = utils.DataIterator(img_dir=args["val_dir"], batch_size=val_batch_size)

    # Variable initialization used for custom training loop
    train_len = len(next(os.walk(args["train_dir"]))[2])
    val_len = len(next(os.walk(args["val_dir"]))[2])
    print("Train Len is", train_len)
    
    if batch_size == 1:
        BATCH_PER_EPOCH = train_len
    else:
        BATCH_PER_EPOCH = int(math.ceil(train_len / batch_size))

    # Initialize tensorboard
    tensorboard = keras.callbacks.TensorBoard(
        log_dir='tmp/my_tf_logs',
        histogram_freq=0,
        write_graph=True
    )

    val_batch_len = int(math.floor(val_len / val_batch_size))
    evaluator = evaluate.Evaluator(val_gen, net, CHARS, val_batch_len, val_batch_size)
    best_val_loss = float("inf")

    # If a pretrained model is available, load weights from it
    if args["pretrained"]:
        net.load_weights(args["pretrained"])

    model = net.model
    tensorboard.set_model(model)

    # Initialize the learning rate
    learning_rate = keras.optimizers.schedules.ExponentialDecay(
        args["lr"],
        decay_steps=args["decay_steps"],
        decay_rate=args["decay_rate"],
        staircase=args["staircase"]
    )

    # Define training optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    print('Training ...')
    train_loss = 0

    # Starting the training loop
    for epoch in range(args["train_epochs"]):
        print(f"Start of epoch {epoch} / {args['train_epochs']}")

        # Zero out the train_loss and val_loss at the beginning of every loop
        train_loss = 0
        val_loss = 0
        start_time = time.time()

        for batch in range(BATCH_PER_EPOCH):
            # Get a batch of images/labels
            # The labels have to be put into sparse tensor to feed into tf.nn.ctc_loss
            train_inputs, train_targets, train_labels = train_gen.next_batch()
            train_inputs = train_inputs.astype('float32')
            train_targets = tf.SparseTensor(train_targets[0], train_targets[1], train_targets[2])

            # Open a GradientTape to record the operations run during the forward pass
            with tf.GradientTape() as tape:
                # Get model outputs
                logits = model(train_inputs, training=True)

                # Pass the model outputs into the ctc loss function
                logits = tf.reduce_mean(logits, axis=1)
                logits_shape = tf.shape(logits)
                cur_batch_size = logits_shape[0]
                timesteps = logits_shape[1]
                seq_len = tf.fill([cur_batch_size], timesteps)
                logits = tf.transpose(logits, (1, 0, 2))
                ctc_loss = tf.nn.ctc_loss(
                    labels=train_targets,
                    logits=logits,
                    label_length=train_labels,
                    logit_length=seq_len,
                    logits_time_major=True,
                    blank_index=-1  # Assuming the blank index is the last class
                )
                loss_value = tf.reduce_mean(ctc_loss)

            # Calculate Gradients and Update it
            grads = tape.gradient(ctc_loss, model.trainable_weights, unconnected_gradients=tf.UnconnectedGradients.NONE)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            train_loss += float(loss_value)

        tim = time.time() - start_time
        print("Train loss {}, time {} \n".format(float(train_loss / BATCH_PER_EPOCH), tim))

        # Run a validation loop every 25 epochs
        if epoch != 0 and epoch % 25 == 0:
            val_loss = evaluator.evaluate()
            # If the validation loss is less than the previous best validation loss, update the saved model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                net.save_weights(os.path.join(args["saved_dir"], "new_out_model_best.keras"))
                print("Weights updated in {}/{}".format(args["saved_dir"], "new_out_model_best.keras"))
            else:
                print("Validation loss is greater than best_val_loss")

    net.save(os.path.join(args["saved_dir"], "new_out_model_last.keras"))
    print("Final Weights saved in {}/{}".format(args["saved_dir"], "new_out_model_last.keras"))
    tensorboard.on_train_end(None)

def parser_args():
    """
    Argument Parser for command line arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_dir", default="/content/LPRNet_ICT/train", help="path to the train directory")
    parser.add_argument("--val_dir", default="/content/LPRNet_ICT/valid", help="path to the validation directory")
    parser.add_argument("--train_epochs", type=int, default=1000, help="number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size (train)")
    parser.add_argument("--val_batch_size", type=int, default=4, help="validation batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="initial learning rate")
    parser.add_argument("--decay_steps", type=float, default=500, help="learning rate decay steps")
    parser.add_argument("--decay_rate", type=float, default=0.995, help="learning rate decay rate")
    parser.add_argument("--staircase", action="store_true", help="learning rate decay on step (default: smooth)")
    parser.add_argument("--pretrained", help="pretrained model location")
    parser.add_argument("--saved_dir", default="/content/LPRNet_ICT/saved_models", help="folder for saving models")

    args = vars(parser.parse_args())
    return args

if __name__ == "__main__":
    args = parser_args()
    tf.compat.v1.enable_eager_execution()
    train()
