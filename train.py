from pathlib import Path
from multiprocessing import Process, Queue

import numpy as np
import tensorflow as tf
import tensorflow.keras as K

import vnet
import data_kits
from config import cfg
import utils

params = {
    "dstRes": np.array([1, 1, 1.5], dtype=float),
    "VolSize": np.array([128, 128, 64], dtype=int),
    "normDir": False  # if rotates the volume according to its transformation in the mhd file. Not reccommended.
}


def load_data():
    dataloader = data_kits.DataManager(cfg.dirTrain, cfg.dirResult, params)
    dataloader.loadTrainingData()
    howManyImages = len(dataloader.sitkImages)
    howManyGT = len(dataloader.sitkGT)
    assert howManyImages == howManyGT
    print(f"The dataset has shape: data - {howManyImages}. labels - {howManyGT}")
    numpyImages = dataloader.getNumpyImages()
    numpyGT = dataloader.getNumpyGT()
    for key in numpyImages:
        fg = numpyImages[key][numpyImages[key] > 0]
        numpyImages[key] -= fg.mean()
        numpyImages[key] /= fg.std()
    return numpyImages, numpyGT


def prepare_data_thread(dataQueue, numpyImages, numpyGT, seed=1234):
    nr_iter = cfg.numIterations
    batchsize = cfg.batchSize

    keysIMG = list(numpyImages.keys())

    nr_iter_dataAug = nr_iter * batchsize
    np.random.seed(seed)
    whichDataList = np.random.randint(len(keysIMG), size=int(nr_iter_dataAug / cfg.nProc))
    whichDataForMatchingList = np.random.randint(len(keysIMG), size=int(nr_iter_dataAug / cfg.nProc))

    for whichData, whichDataForMatching in zip(whichDataList, whichDataForMatchingList):
        path = keysIMG[whichData]

        currGtKey = path.parent / (path.stem + '_segmentation.mhd')
        currImgKey = path

        # data agugumentation through hist matching across different examples...
        ImgKeyMatching = keysIMG[whichDataForMatching]

        defImg = numpyImages[currImgKey]
        defLab = numpyGT[currGtKey]

        defImg = data_kits.hist_match(defImg, numpyImages[ImgKeyMatching])

        if np.random.rand(1)[0] > 0.5:  # do not apply deformations always, just sometimes
            defImg, defLab = data_kits.produce_randomly_deformed_image(
                defImg, defLab, cfg.numcontrolpoints, cfg.sigma)

        defImg = defImg.astype(np.float32)
        defLab = (defLab > 0.5).astype(np.int32)

        # weightData = np.zeros_like(defLab, dtype=float)
        # weightData[defLab == 1] = np.prod(defLab.shape) / np.sum((defLab == 1).astype(dtype=np.float32))
        # weightData[defLab == 0] = np.prod(defLab.shape) / np.sum((defLab == 0).astype(dtype=np.float32))

        dataQueue.put(tuple((defImg[..., None], defLab, None)))


def train():
    # Load data and prepare training samples
    numpyImages, numpyGT = load_data()
    dataQueue = Queue(30)  # max 50 images in queue
    dataPreparation = [None] * cfg.nProc

    # thread creation
    for proc in range(cfg.nProc):
        dataPreparation[proc] = Process(target=prepare_data_thread, args=(dataQueue, numpyImages, numpyGT))
        dataPreparation[proc].daemon = True
        dataPreparation[proc].start()

    def data_gen():
        for _ in range(cfg.numIterations * cfg.batchSize):
            defImg, defLab, _ = dataQueue.get()
            yield defImg, defLab

    print("Load data.")
    # tensorflow data loader
    h, w, d = params["VolSize"]
    dataset = tf.data.Dataset.from_generator(data_gen, (tf.float32, tf.int32),
                                             (tf.TensorShape([h, w, d, 1]), tf.TensorShape([h, w, d])))
    dataset = dataset.batch(batch_size=cfg.batchSize)

    print("Build model.")
    # build model
    model = vnet.VNet([h, w, d, 1], cfg.batchSize, cfg.ncls)
    learning_rate = cfg.baseLR
    learning_rate = K.optimizers.schedules.ExponentialDecay(learning_rate, cfg.decay_steps, cfg.decay_rate, True)
    optim = K.optimizers.SGD(learning_rate, momentum=0.99)
    criterion = K.losses.SparseCategoricalCrossentropy(from_logits=True)

    @tf.function
    def train_step(x, y):
        # Forward
        with tf.GradientTape() as tape:
            prediction = model(x)
            losses = criterion(y, prediction)
        # Backward
        with tf.name_scope("Gradients"):
            gradients = tape.gradient(losses, model.trainable_variables)
        optim.apply_gradients(zip(gradients, model.trainable_variables))
        return losses, prediction

    # File writer
    writer, logdir = utils.summary_writer(cfg)
    # Trace graph
    tf.summary.trace_on(graph=True)
    train_step(tf.zeros([1, h, w, d, 1]), tf.zeros([1, h, w, d]))  # dry run for tracing graph (step=1)
    tf.summary.trace_export("OpGraph", 0)

    print("Start training.")
    save_path = logdir / "snapshots"
    total_loss = 0
    dice = None
    for trImg, trLab in dataset:
        loss, pred = train_step(trImg, trLab)
        step = optim.iterations.numpy()     # (step start from 2)
        loss_val = loss.numpy()

        # Loss moving average
        total_loss = loss_val if step < 5 else \
            cfg.moving_average * total_loss + (1 - cfg.moving_average) * loss_val

        # Logging
        if (step < 500 and step % 10 == 0) or step % cfg.log_interval == 0:
            dice = utils.compute_dice(trLab, pred)
            print(f"Step: {step}, Loss: {loss_val:.4f}, Dice: {dice:.4f}, "
                  f"LR: {learning_rate(step).numpy():.2E}")

            # Summary scalars and images
            tf.summary.scalar("loss", total_loss, step=step)
            tf.summary.scalar("dice", dice, step=step)
            tf.summary.image("trImg", trImg[..., d // 2, :], step=step)
            tf.summary.image("pred", pred[..., d // 2, :], step=step)

        # Take snapshots
        if step == 2 or step % cfg.snap_shot_interval == 0:
            filepath = utils.snapshot(model, save_path, step)
            print(f"Model weights saved (Path: {filepath}).")

    # Ending
    filepath = utils.snapshot(model, save_path, optim.iterations.numpy())
    print(f"Model weights saved ({filepath}).\nTraining ended.")
    writer.close()


if __name__ == "__main__":
    train()
