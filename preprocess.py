import pandas as pd
import numpy as np
import scipy.sparse as sp
from optparse import OptionParser
import config

def preprocess(data_name):
    if data_name == "wn18":
        raw_train_file = config.WN18_TRAIN_RAW
        raw_valid_file = config.WN18_VALID_RAW
        raw_test_file = config.WN18_TEST_RAW
        train_file = config.WN18_TRAIN
        valid_file = config.WN18_VALID
        test_file = config.WN18_TEST
        e2id_file = config.WN18_E2ID
        r2id_file = config.WN18_R2ID
        sub_mat_file = config.WN18_SUB_MAT
        obj_mat_file = config.WN18_OBJ_MAT
    elif data_name == "fb15k":
        raw_train_file = config.FB15K_TRAIN_RAW
        raw_valid_file = config.FB15K_VALID_RAW
        raw_test_file = config.FB15K_TEST_RAW
        train_file = config.FB15K_TRAIN
        valid_file = config.FB15K_VALID
        test_file = config.FB15K_TEST
        e2id_file = config.FB15K_E2ID
        r2id_file = config.FB15K_R2ID
    elif data_name == "fb1m":
        raw_train_file = config.FB1M_TRAIN_RAW
        raw_valid_file = config.FB1M_VALID_RAW
        raw_test_file = config.FB1M_TEST_RAW
        train_file = config.FB1M_TRAIN
        valid_file = config.FB1M_VALID
        test_file = config.FB1M_TEST
        e2id_file = config.FB1M_E2ID
        r2id_file = config.FB1M_R2ID

    df_train = pd.read_csv(raw_train_file, sep="\t", names=["e1", "r", "e2"])
    df_valid = pd.read_csv(raw_valid_file, sep="\t", names=["e1", "r", "e2"])
    df_test = pd.read_csv(raw_test_file, sep="\t", names=["e1", "r", "e2"])
    df_all = pd.concat([df_train, df_valid, df_test], ignore_index=True)
    train_size = df_train.shape[0]
    valid_size = df_valid.shape[0]
    test_size = df_test.shape[0]

    outfile = open(e2id_file, "w")
    e2id = {}
    for idx, e in enumerate(set(list(df_all.e1) + list(df_all.e2))):
        e2id[e] = idx
        outfile.write("%s %d\n" % (e, idx))
    outfile.close()

    outfile = open(r2id_file, "w")
    r2id = {}
    for idx, r in enumerate(set(list(df_all.r))):
        r2id[r] = idx
        outfile.write("%s %d\n" % (r, idx))
    outfile.close()

    df_all.e1 = df_all.e1.map(e2id)
    df_all.e2 = df_all.e2.map(e2id)
    df_all.r = df_all.r.map(r2id)

    data = np.ones(train_size)
    row = df_all.r[:train_size]
    col1 = df_all.e1[:train_size]
    col2 = df_all.e2[:train_size]
    sub_mat = sp.coo_matrix((data, (row, col1)), shape=(len(r2id), len(e2id)))
    obj_mat = sp.coo_matrix((data, (row, col2)), shape=(len(r2id), len(e2id)))
    sub_mat = sub_mat.todok().tocoo()
    obj_mat = obj_mat.todok().tocoo()

    df_all[:train_size].to_csv(train_file, header=False, index=False)
    df_all[train_size:train_size + valid_size].to_csv(valid_file, header=False, index=False)
    df_all[train_size + valid_size:].to_csv(test_file, header=False, index=False)
    sp.save_npz(sub_mat_file, sub_mat)
    sp.save_npz(obj_mat_file, obj_mat)

def parse_args(parser):
    parser.add_option("-d", "--data", type="string", dest="data_name", default="wn18")

    options, args = parser.parse_args()
    return options, args

def main(options):
    data_name = options.data_name
    preprocess(data_name)

if __name__ == "__main__":
    parser = OptionParser()
    options, args = parse_args(parser)
    main(options)
