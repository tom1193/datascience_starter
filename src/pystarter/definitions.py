import os
from dotenv import load_dotenv
load_dotenv()

ROOT = "/home/local/datascience_starter"
DATA = os.path.join(ROOT, "data")
LOGS = os.path.join(ROOT, "logs")
CHECKPOINTS = os.path.join(ROOT, "checkpoints")

RANDOM_SEED = 101

SALT = os.environ.get('SALT')