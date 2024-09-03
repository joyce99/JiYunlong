import numpy as np
import tensorflow as tf
import scipy.io as sio
from scipy.sparse import csc_matrix
from scipy.sparse import lil_matrix
import scipy
from sklearn.metrics import f1_score
import scipy.sparse as sp
from sklearn.decomposition import PCA


def shuffle_aligned_list(data):
    num = data[0].shape[0]
    shuffle_index = np.random.permutation(num)  # 随机重排
    return shuffle_index, [d[shuffle_index] for d in data]


def csr_2_sparse_tensor_tuple(csr_matrix):
    if not isinstance(csr_matrix, scipy.sparse.lil_matrix):
        csr_matrix = lil_matrix(csr_matrix)
    coo_matrix = csr_matrix.tocoo()
    indices = np.transpose(np.vstack((coo_matrix.row, coo_matrix.col)))  # 转置
    values = coo_matrix.data
    shape = csr_matrix.shape
    return indices, values, shape


def batch_generator(data, batch_size, shuffle=True):
    if shuffle:
        shuffle_index, data = shuffle_aligned_list(data)
    batch_count = 0
    while True:
        if batch_count * batch_size + batch_size >= data[0].shape[0]:
            batch_count = 0
            if shuffle:
                shuffle_index, data = shuffle_aligned_list(data)
        start = batch_count * batch_size
        end = start + batch_size
        batch_count += 1
        yield [d[start:end] for d in data], shuffle_index[start:end]


def batch_generator_shunxu(data1, data2, batchsize, startindex):
    size = min(batchsize, data1.shape[0] - startindex)
    batch_att1 = data1[startindex:startindex + size]
    batch_att2 = data2[startindex:startindex + size]
    if startindex + batchsize > data1.shape[0]:
        startindex = 0
    else:
        startindex = startindex + batchsize
    return batch_att1, batch_att2, startindex


def batch_generator_shunxu1(data, batchsize, startindex):
    size = min(batchsize, data.shape[0] - startindex)
    batch_att = data[startindex:startindex + size]

    if startindex + batchsize > data.shape[0]:
        startindex = 0
    else:
        startindex = startindex + batchsize
    return batch_att, startindex


def fc_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu, input_type='dense', drop=0.0):
    with tf.name_scope(layer_name):
        weight = tf.Variable(tf.truncated_normal([input_dim, output_dim], stddev=1. / tf.sqrt(input_dim / 2.)),
                             name='weight')
        bias = tf.Variable(tf.constant(0.1, shape=[output_dim]), name='bias')
        if input_type == 'sparse':
            activations = act(tf.sparse_tensor_dense_matmul(input_tensor, weight) + bias)
        else:
            activations = act(tf.matmul(input_tensor, weight) + bias)

        activations = tf.nn.dropout(activations, rate=drop)

        return activations


def load_network(file):
    net = sio.loadmat(file)
    X, A, Y = net['attrb'], net['network'], net['group']
    if not isinstance(X, scipy.sparse.lil_matrix):
        X = lil_matrix(X)

    return A, X, Y


def load_network1(file):
    net = sio.loadmat(file, mat_dtype=True)
    X, A, Y = net['attrb'], net['network'], net['group']
    if not isinstance(X, scipy.sparse.lil_matrix):
        X = lil_matrix(X)

    return A, X, Y


def MyScaleSimMat(W):
    '''L1 row norm of a matrix'''
    rowsum = np.array(np.sum(W, axis=1), dtype=np.float32)
    rowsum[rowsum == 0] = 1e-10  # 将 rowsum 中为 0 的元素替换为 1e-10，后加的
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    W = r_mat_inv.dot(W)
    return W


def AggTranProbMat(G, step):
    '''aggregated K-step transition probality'''
    G = MyScaleSimMat(G)
    G = csc_matrix.toarray(G)
    A_k = G
    A = G
    for k in np.arange(2, step + 1):
        A_k = np.matmul(A_k, G)
        A = A + A_k / k

    return A


def ComputePPMI(A):
    '''compute PPMI, given aggregated K-step transition probality matrix as input'''
    np.fill_diagonal(A, 0)
    A = MyScaleSimMat(A)
    (p, q) = np.shape(A)
    col = np.sum(A, axis=0)
    col[col == 0] = 1
    PPMI = np.log((float(p) * A) / col[None, :])
    IdxNan = np.isnan(PPMI)
    PPMI[IdxNan] = 0
    PPMI[PPMI < 0] = 0

    return PPMI


def batchPPMI(batch_size, shuffle_index_s, shuffle_index_t, PPMI_s, PPMI_t):
    '''return the PPMI matrix between nodes in each batch'''

    ##proximity matrix between source network nodes in each mini-batch
    a_s = np.zeros((int(batch_size / 2), int(batch_size / 2)))
    for ii in range(int(batch_size / 2)):
        for jj in range(int(batch_size / 2)):
            if ii != jj:
                a_s[ii, jj] = PPMI_s[shuffle_index_s[ii], shuffle_index_s[jj]]

    ##proximity matrix between target network nodes in each mini-batch
    a_t = np.zeros((int(batch_size / 2), int(batch_size / 2)))
    for ii in range(int(batch_size / 2)):
        for jj in range(int(batch_size / 2)):
            if ii != jj:
                a_t[ii, jj] = PPMI_t[shuffle_index_t[ii], shuffle_index_t[jj]]

    return csr_2_sparse_tensor_tuple(MyScaleSimMat(a_s)), csr_2_sparse_tensor_tuple(MyScaleSimMat(a_t))


def batchSim(batch_size, shuffle_index_s, shuffle_index_t, att_sim_s, att_sim_t):
    '''return the similarity matrix between nodes in each batch'''

    ##proximity matrix between source network nodes in each mini-batch
    att_s = np.zeros((int(batch_size / 2), int(batch_size / 2)))
    for ii in range(int(batch_size / 2)):
        for jj in range(int(batch_size / 2)):
            att_s[ii, jj] = att_sim_s[shuffle_index_s[ii], shuffle_index_s[jj]]

    ##proximity matrix between target network nodes in each mini-batch
    att_t = np.zeros((int(batch_size / 2), int(batch_size / 2)))
    for ii in range(int(batch_size / 2)):
        for jj in range(int(batch_size / 2)):
            att_t[ii, jj] = att_sim_t[shuffle_index_t[ii], shuffle_index_t[jj]]

    return csr_2_sparse_tensor_tuple(MyScaleSimMat(att_s)), csr_2_sparse_tensor_tuple(MyScaleSimMat(att_t))


def feature_compression(features, dim=200):
    """Preprcessing of features"""
    features = features.toarray()
    feat = lil_matrix(PCA(n_components=dim, random_state=0).fit_transform(features))
    return feat


def f1_scores(y_pred, y_true):
    def predict(y_true, y_pred):
        top_k_list = np.array(np.sum(y_true, 1), np.int32)
        predictions = []
        for i in range(y_true.shape[0]):
            pred_i = np.zeros(y_true.shape[1])
            pred_i[np.argsort(y_pred[i, :])[-top_k_list[i]:]] = 1
            predictions.append(np.reshape(pred_i, (1, -1)))
        predictions = np.concatenate(predictions, axis=0)

        return np.array(predictions, np.int32)

    results = {}
    predictions = predict(y_true, y_pred)
    averages = ["micro", "macro"]
    for average in averages:
        results[average] = f1_score(y_true, predictions, average=average)

    return results["micro"], results["macro"]


def euclidean_distance(x, y):
    return np.sqrt(np.sum((x - y) ** 2))


def pairwise_euclidean_distance(matrix1, matrix2):
    distance_matrix = np.zeros((matrix1.shape[0], matrix2.shape[0]))
    for i, row1 in enumerate(matrix1):
        for j, row2 in enumerate(matrix2):
            distance_matrix[i, j] = euclidean_distance(row1, row2)
    return distance_matrix


def manhattan_distance(x, y):
    return np.sum(np.abs(x - y))


def pairwise_manhattan_distance(matrix1, matrix2):
    distance_matrix = np.zeros((matrix1.shape[0], matrix2.shape[0]))
    for i, row1 in enumerate(matrix1):
        for j, row2 in enumerate(matrix2):
            distance_matrix[i, j] = manhattan_distance(row1, row2)
    return distance_matrix


def cosine_distance(vector1, vector2):
    dot_product = np.dot(vector1, np.transpose(vector2))
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)
    similarity = dot_product / (norm1 * norm2)
    distance = 1 - similarity
    return distance


def predictgetone(y_true, y_pred):
    top_k_list = np.array(np.sum(y_true, 1), np.int32)
    predictions = []
    for i in range(y_true.shape[0]):
        pred_i = np.zeros(y_true.shape[1])
        pred_i[np.argsort(y_pred[i, :])[-top_k_list[i]:]] = 1
        predictions.append(np.reshape(pred_i, (1, -1)))
    predictions = np.concatenate(predictions, axis=0)

    return np.array(predictions, np.int32)


def f1_known_score(y_true, predictions):
    results = {}

    averages = ["micro", "macro"]
    for average in averages:
        results[average] = f1_score(y_true, predictions, average=average)

    return results["micro"], results["macro"]