# ----------------------- PATH ------------------------

ROOT_PATH = "."
DATA_PATH = "%s/data" % ROOT_PATH
WN18_DATA_PATH = "%s/wn18" % DATA_PATH
WN18RR_DATA_PATH = "%s/wn18rr" % DATA_PATH
FB15K_DATA_PATH = "%s/fb15k" % DATA_PATH
FB15K237_DATA_PATH = "%s/fb15k-237" % DATA_PATH
FB1M_DATA_PATH = "%s/fb1m" % DATA_PATH

LOG_PATH = "%s/log" % ROOT_PATH
CHECKPOINT_PATH = "%s/checkpoint" % ROOT_PATH

# ----------------------- DATA ------------------------

WN18_TRAIN_RAW = "%s/train.txt" % WN18_DATA_PATH
WN18_VALID_RAW = "%s/valid.txt" % WN18_DATA_PATH
WN18_TEST_RAW = "%s/test.txt" % WN18_DATA_PATH

WN18_TRAIN = "%s/digitized_train.txt" % WN18_DATA_PATH
WN18_VALID = "%s/digitized_valid.txt" % WN18_DATA_PATH
WN18_TEST = "%s/digitized_test.txt" % WN18_DATA_PATH

WN18_E2ID = "%s/e2id.txt" % WN18_DATA_PATH
WN18_R2ID = "%s/r2id.txt" % WN18_DATA_PATH
WN18_SUB_MAT = "%s/sub_mat.npz" % WN18_DATA_PATH
WN18_OBJ_MAT = "%s/obj_mat.npz" % WN18_DATA_PATH

WN18RR_TRAIN_RAW = "%s/train.txt" % WN18RR_DATA_PATH
WN18RR_VALID_RAW = "%s/valid.txt" % WN18RR_DATA_PATH
WN18RR_TEST_RAW = "%s/test.txt" % WN18RR_DATA_PATH

WN18RR_TRAIN = "%s/digitized_train.txt" % WN18RR_DATA_PATH
WN18RR_VALID = "%s/digitized_valid.txt" % WN18RR_DATA_PATH
WN18RR_TEST = "%s/digitized_test.txt" % WN18RR_DATA_PATH

WN18RR_E2ID = "%s/e2id.txt" % WN18RR_DATA_PATH
WN18RR_R2ID = "%s/r2id.txt" % WN18RR_DATA_PATH
WN18RR_SUB_MAT = "%s/sub_mat.npz" % WN18RR_DATA_PATH
WN18RR_OBJ_MAT = "%s/obj_mat.npz" % WN18RR_DATA_PATH

FB15K_TRAIN_RAW = "%s/train.txt" % FB15K_DATA_PATH
FB15K_VALID_RAW = "%s/valid.txt" % FB15K_DATA_PATH
FB15K_TEST_RAW = "%s/test.txt" % FB15K_DATA_PATH

FB15K_TRAIN = "%s/digitized_train.txt" % FB15K_DATA_PATH
FB15K_VALID = "%s/digitized_valid.txt" % FB15K_DATA_PATH
FB15K_TEST = "%s/digitized_test.txt" % FB15K_DATA_PATH

FB15K_E2ID = "%s/e2id.txt" % FB15K_DATA_PATH
FB15K_R2ID = "%s/r2id.txt" % FB15K_DATA_PATH
FB15K_SUB_MAT = "%s/sub_mat.npz" % FB15K_DATA_PATH
FB15K_OBJ_MAT = "%s/obj_mat.npz" % FB15K_DATA_PATH

FB15K237_TRAIN_RAW = "%s/train.txt" % FB15K237_DATA_PATH
FB15K237_VALID_RAW = "%s/valid.txt" % FB15K237_DATA_PATH
FB15K237_TEST_RAW = "%s/test.txt" % FB15K237_DATA_PATH

FB15K237_TRAIN = "%s/digitized_train.txt" % FB15K237_DATA_PATH
FB15K237_VALID = "%s/digitized_valid.txt" % FB15K237_DATA_PATH
FB15K237_TEST = "%s/digitized_test.txt" % FB15K237_DATA_PATH

FB15K237_E2ID = "%s/e2id.txt" % FB15K237_DATA_PATH
FB15K237_R2ID = "%s/r2id.txt" % FB15K237_DATA_PATH
FB15K237_SUB_MAT = "%s/sub_mat.npz" % FB15K237_DATA_PATH
FB15K237_OBJ_MAT = "%s/obj_mat.npz" % FB15K237_DATA_PATH

FB1M_TRAIN_RAW = "%s/train.txt" % FB1M_DATA_PATH
FB1M_VALID_RAW = "%s/valid.txt" % FB1M_DATA_PATH
FB1M_TEST_RAW = "%s/test.txt" % FB1M_DATA_PATH

FB1M_TRAIN = "%s/digitized_train.txt" % FB1M_DATA_PATH
FB1M_VALID = "%s/digitized_valid.txt" % FB1M_DATA_PATH
FB1M_TEST = "%s/digitized_test.txt" % FB1M_DATA_PATH

FB1M_E2ID = "%s/e2id.txt" % FB1M_DATA_PATH
FB1M_R2ID = "%s/r2id.txt" % FB1M_DATA_PATH

# ----------------------- PARAM -----------------------

RANDOM_SEED = None
