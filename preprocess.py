import pandas as pd
from optparse import OptionParser
import config
from utils import data_utils

def preprocess(data_name):
    dataset = data_utils.DataSet(config.DATASET[data_name])

    df_train, df_valid, df_test = dataset.load_raw_data()
    df_all = pd.concat([df_train, df_valid, df_test], ignore_index=True)
    train_size = df_train.shape[0]
    test_size = df_test.shape[0]

    e2id = dataset.save_e2id(set(list(df_all.e1) + list(df_all.e2)))
    r2id = dataset.save_r2id(set(list(df_all.r)))

    df_all.e1 = df_all.e1.map(e2id)
    df_all.e2 = df_all.e2.map(e2id)
    df_all.r = df_all.r.map(r2id)

    # data = np.ones(train_size)
    # row = df_all.r[:train_size]
    # col1 = df_all.e1[:train_size]
    # col2 = df_all.e2[:train_size]
    # sub_mat = sp.coo_matrix((data, (row, col1)), shape=(len(r2id), len(e2id)))
    # obj_mat = sp.coo_matrix((data, (row, col2)), shape=(len(r2id), len(e2id)))
    # sub_mat = sub_mat.todok().tocoo()
    # obj_mat = obj_mat.todok().tocoo()

    # embedding_mat = np.random.uniform(-1, 1, (len(e2id), model.syn0.shape[1]))
    # for e in e2id:
    #    if e in model:
    #        embedding_mat[e2id[e]] = model[e]
    # feat_mat = np.hstack((sub_mat.todense().T, obj_mat.todense().T, embedding_mat))

    df_train = df_all[:train_size]
    df_valid = df_all[train_size:-test_size]
    df_test = df_all[-test_size:]
    dataset.save_data(df_train, df_valid, df_test)

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
