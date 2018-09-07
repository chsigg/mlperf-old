# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Wrapper scripts to ensure that main.py commands are called correctly."""
import argh
import argparse
import cloud_logging
import logging
import os
import shipname
import sys
import time
import shutil
import dual_net
import preprocessing
import subprocess

import glob
from tensorflow import gfile

from utils import timer
import logging

import goparams
import predict_moves

import qmeas

SEED = None
ITERATION = None

# Pull in environment variables. Run `source ./cluster/common` to set these.
#BUCKET_NAME = os.environ['BUCKET_NAME']

#BASE_DIR = "gs://{}".format(BUCKET_NAME)
BASE_DIR = goparams.BASE_DIR

MODELS_DIR = os.path.join(BASE_DIR, 'models')
SELFPLAY_DIR = os.path.join(BASE_DIR, 'data/selfplay')
BURY_DIR = os.path.join(BASE_DIR, 'bury_models')
HOLDOUT_DIR = os.path.join(BASE_DIR, 'data/holdout')
SGF_DIR = os.path.join(BASE_DIR, 'sgf')
TRAINING_CHUNK_DIR = os.path.join(BASE_DIR, 'data', 'training_chunks')

ESTIMATOR_WORKING_DIR = os.path.join(BASE_DIR, 'estimator_working_dir')

# How many games before the selfplay workers will stop trying to play more.
MAX_GAMES_PER_GENERATION = goparams.MAX_GAMES_PER_GENERATION

# What percent of games to holdout from training per generation

HOLDOUT_PCT = goparams.HOLDOUT_PCT


def print_flags():
    flags = {
        #'BUCKET_NAME': BUCKET_NAME,
        'BASE_DIR': BASE_DIR,
        'MODELS_DIR': MODELS_DIR,
        'SELFPLAY_DIR': SELFPLAY_DIR,
        'HOLDOUT_DIR': HOLDOUT_DIR,
        'SGF_DIR': SGF_DIR,
        'TRAINING_CHUNK_DIR': TRAINING_CHUNK_DIR,
        'ESTIMATOR_WORKING_DIR': ESTIMATOR_WORKING_DIR,
    }
    print("Computed variables are:")
    print('\n'.join('--{}={}'.format(flag, value)
                    for flag, value in flags.items()))


def get_models():
    """Finds all models, returning a list of model number and names
    sorted increasing.

    Returns: [(13, 000013-modelname), (17, 000017-modelname), ...etc]
    """
    all_models = gfile.Glob(os.path.join(MODELS_DIR, '*.meta'))
    model_filenames = [os.path.basename(m) for m in all_models]
    model_numbers_names = sorted([
        (shipname.detect_model_num(m), shipname.detect_model_name(m))
        for m in model_filenames])
    return model_numbers_names


def get_latest_model():
    """Finds the latest model, returning its model number and name

    Returns: (17, 000017-modelname)
    """
    models = get_models()
    if len(models) == 0:
        models = [(0, '000000-bootstrap')]
    return models[-1]



def main_():
    """Run the reinforcement learning loop

    This tries to create a realistic way to run the reinforcement learning with
    all default parameters.
    """
    print('Starting self play loop.')

    qmeas.start_time('selfplay_wait')
    start_t = time.time()

    _, model_name = get_latest_model()

    # TODO: remove NUM_PARALLEL_SELFPLAY from goparams
    cmd = [
      "./external/com_google_minigo/cc/main",
      "--mode=selfplay",
      "--model={}".format(os.path.join(MODELS_DIR, model_name)),
      "--parallel_games={}".format(MAX_GAMES_PER_GENERATION),
      "--num_readouts={}".format(goparams.SP_READOUTS),

      "--output_dir={}".format(os.path.join(SELFPLAY_DIR, model_name)),
      "--holdout_dir={}".format(os.path.join(HOLDOUT_DIR, model_name)),
      "--holdout_pct={0:f}".format(HOLDOUT_PCT),
      "--sgf_dir={}".format(os.path.join(SGF_DIR, model_name)),
      "--subdir_format=",

      "--virtual_losses=8" # see SIMULTANEOUS_LEAVES,
    ]
    print("Running", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)

    qmeas.stop_time('selfplay_wait')


if __name__ == '__main__':
    #tf.logging.set_verbosity(tf.logging.INFO)
    qmeas.start(os.path.join(BASE_DIR, 'stats'))

    SEED = int(sys.argv[1])
    ITERATION = int(sys.argv[2])

    # get TF logger
    log = logging.getLogger('tensorflow')
    log.setLevel(logging.DEBUG)

    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # create file handler which logs even debug messages
    fh = logging.FileHandler('tensorflow.log')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    log.addHandler(fh)
    main_()
    qmeas.end()
