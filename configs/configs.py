DATA_DIR = "./data"
BATCH_SIZE = 64  # 1  # stochastic
NUM_WORKERS = 0

LR_SCHEDULE = [  # (# of passes, lr) total 20 passes
    (2, 0.0005),
    (3, 0.0002),
    (3, 0.0001),
    (4, 0.00005),
    (8, 0.00001),
]

MU = 0.02  # step size safety factor
HESSIAN_SAMPLE_SIZE = 50  # 500  # Num of samples for Hessian re-estimation
J = 0.1  # incorrect class penalty in loss function