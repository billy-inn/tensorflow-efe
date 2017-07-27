import numpy as np
import pandas as pd
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
import config
from optparse import OptionParser

class Task:
	def __init__(self, ):
