import os
import sys
import logging
import numpy as np
from datetime import datetime
import requests

# Logging
FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(format=FORMAT, level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)

logger.info("EntEval started")

# Set PATH to EntEval and import EntEval
PATH_TO_ENTEVAL = '.'
sys.path.insert(0, PATH_TO_ENTEVAL)
try:
    import enteval
except ImportError:
    logger.warning("EntEval import failed")


# Get envs
try:
    PATH_TO_DATA =  os.getenv('PATH_TO_DATA')
    PATH_TO_RESULTS = os.getenv('PATH_TO_RESULTS')
    ENCODER_URL = os.getenv('ENCODER_URL')
except KeyError as error:
    logger.warning("Env Variable not found: {}".format(error))


def prepare(params, batch):
    pass

def batcher(params, batch):
    use_ctx = False
    use_def = False
    if batch[0][0] is not None:
        use_ctx = True
    if batch[0][3] is not None:
        use_def = True

    if use_ctx:
        max_context_len = max([len(item[0]) for item in batch])
        batch_contexts = []
        batch_context_span_mask = np.zeros((len(batch), max_context_len)).astype("float32")
    if use_def:
        batch_desc = []

    for i, (ctx, s, e, desc) in enumerate(batch):
        if use_ctx:
            batch_contexts.append([w.lower() for w in ctx] if ctx != [] else ['.'])
            batch_context_span_mask[i, s:] = 1.
            batch_context_span_mask[i, e:] = 0.
        if use_def:
            batch_desc.append([w.lower() for w in desc] if desc != [] else ['.'])

    ctx_embedding = np.zeros(len(batch)).astype("float32")
    desc_embedding = np.zeros(len(batch)).astype("float32")
    #logger.info("Ctx embedding shape: {}".format(ctx_embedding.shape))

    if use_ctx:
        r = requests.post(ENCODER_URL, json=batch_contexts)
        ctx_embedding = np.array(r.json())#[0])

        #logger.info("Ctx Embedding shape: {}".format(ctx_embedding.shape))

    return ctx_embedding, desc_embedding

# Set params for EntEval
params_enteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'batch_size':
        8}
params_enteval['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64, 'tenacity': 5, 'epoch_size': 4}


def evaluate():
    task = 2
    layer = 1
    
    task_groups = [
        ["CAPsame", "CAPnext", "CERP", "EFP", "KORE", "WikiSRS", "ERT"], 
        ["Rare"], 
        ["ET"], 
        ["ConllYago"]
    ]

    tasks = ["ET", "CAPsame", "CAPnext", "EFP", "CERP"]

    logger.info("Starting evaluation for task: {}".format(tasks))

    se = enteval.engine.SE(params_enteval, batcher, prepare)
    results = se.eval(tasks)

    logger.info("Done evaluating")
    return results




if __name__ == '__main__':
    results = evaluate()

    # filename for results
    now = datetime.now().strftime("%d-%m-%Y-%H_%M_%S")
    filename = "EntEval_" + now + ".txt"
    file_path = os.path.join(PATH_TO_RESULTS, filename)

    # write results
    logger.info("Writing results to {}".format(file_path))
    try:
        with open(file_path, "w") as f:
            for k, v in results.items():
                f.write("{}, {}".format(k,v))
                f.write("\n")
        logger.info("Wrote results to {}".format(filename))

    except IOError as error:
        logger.warning("Could not open {}. {}".format(file_path, error))
