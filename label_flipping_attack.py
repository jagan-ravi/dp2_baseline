from federated_learning.worker_selection import RandomSelectionStrategy
from server import run_exp
from functions import round_worker

if __name__ == '__main__':
    START_EXP_IDX = 901 ####Always Start with oned Index eg:3001
    NUM_EXP = 1
    NUM_POISONED_WORKERS=0
    NUM_WORKERS_PER_ROUND=1
    NUM_OF_REPLACEMENT=0
    LABELS_TO_REPLACE=[]
    LABELS_TO_REPLACE_WITH=[]
    PERCENTAGE_OF_REPLACEMENT=0
    ROUNDS=2
    NUM_WORKERS=1
    RANDOM_WORKERS=round_worker(ROUNDS,NUM_WORKERS,NUM_WORKERS_PER_ROUND)

    KWARGS = {
        "EXPID":0,
        "ROUNDS":ROUNDS,
        "NUM_WORKERS":NUM_WORKERS,
        "NUM_EXP":NUM_EXP,
        "NUM_POISONED_WORKERS":NUM_POISONED_WORKERS,
        "NUM_WORKERS_PER_ROUND" : NUM_WORKERS_PER_ROUND,
        "RANDOM_WORKERS":RANDOM_WORKERS,
        "NUM_OF_REPLACEMENT":NUM_OF_REPLACEMENT,
        "LABELS_TO_REPLACE":LABELS_TO_REPLACE,
        "LABELS_TO_REPLACE_WITH":LABELS_TO_REPLACE_WITH,
        "PERCENTAGE_OF_REPLACEMENT":PERCENTAGE_OF_REPLACEMENT
    }
    for experiment_id in range(START_EXP_IDX, START_EXP_IDX + NUM_EXP):
        run_exp(KWARGS, RandomSelectionStrategy(), experiment_id)
