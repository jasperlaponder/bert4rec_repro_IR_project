from aprec.recommenders.vanilla_bert4rec import VanillaBERT4Rec
from aprec.evaluation.metrics.hit import HIT
from aprec.evaluation.metrics.recall import Recall
from aprec.evaluation.split_actions import RandomSplit
import numpy as np


DATASET = "BERT4rec.dunnhumby"

USERS_FRACTIONS = [1]

def original_bert4rec(training_steps):
    recommender = VanillaBERT4Rec(num_train_steps=training_steps)
    return recommender

RECOMMENDERS = {
    "Original-bert4rec-400000": lambda: original_bert4rec(400000),
}

MAX_TEST_USERS=6040
TEST_FRACTION=0.2

METRICS = [Recall(10), HIT(10), Recall(20), HIT(20)]

RECOMMENDATIONS_LIMIT = 100
SPLIT_STRATEGY = RandomSplit(TEST_FRACTION, MAX_TEST_USERS)

