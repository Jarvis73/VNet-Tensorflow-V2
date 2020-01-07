class cfg(object):
    numcontrolpoints = 2
    sigma = 15
    gpu = 0

    tag = "exp1"
    dirTrain = "./data/train"
    dirTest = "./data/test"
    dirResult = "./data/result"     # where we need to save the results (relative to the base path)
    dirRuns = "./runs"              # where to save the models while training

    ncls = 2                        # the number of classes
    batchSize = 2                   # the batchsize
    numIterations = 100000          # the number of iterations
    baseLR = 0.0001                 # the learning rate, initial one
    decay_steps = 20000
    decay_rate = 0.1
    nProc = 2                       # the number of threads to do data augmentation

    log_interval = 100
    moving_average = 0.95
    snap_shot_interval = 10000
