import warnings

warnings.filterwarnings('ignore')
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import utils
from scipy.sparse import vstack
from evalModel import train_and_evaluate
from scipy.sparse import lil_matrix
from sklearn.metrics.pairwise import cosine_similarity
import recon_process

tf.set_random_seed(0)
np.random.seed(0)


source = 'Blog1'
target = 'Blog2'
emb_filename = str(source) + '_' + str(target)
dirname = str(source) + '-' + str(target)
Kstep = 3
random_state=0
print('source and target networks:', str(source), str(target))

A_s, X_s, Y_s = utils.load_network('./data/' + dirname + '/' + str(source) + '.mat')
num_nodes_S = X_s.shape[0]
num_class_S = Y_s.shape[1]
num_node_s_old=X_s.shape[0]


A_t, X_t, Y_t = utils.load_network('./data/' + dirname + '/' + str(target) + '.mat')
num_nodes_T = X_t.shape[0]

X_s, Y_s, A_s, add_total = recon_process.recon(X_s, Y_s, A_s, X_t, A_t,Y_t)


features = vstack((X_s, X_t))
features = utils.feature_compression(features, dim=1000)
X_s = features[0:num_nodes_S + add_total, :]

X_t = features[-num_nodes_T:, :]

'''compute PPMI'''
A_k_s = utils.AggTranProbMat(A_s, Kstep)
PPMI_s = utils.ComputePPMI(A_k_s)
n_PPMI_s = utils.MyScaleSimMat(PPMI_s)
X_n_s = np.matmul(n_PPMI_s, lil_matrix.toarray(X_s))

'''compute PPMI'''
A_k_t = utils.AggTranProbMat(A_t, Kstep)
PPMI_t = utils.ComputePPMI(A_k_t)
n_PPMI_t = utils.MyScaleSimMat(PPMI_t)
X_n_t = np.matmul(n_PPMI_t, lil_matrix.toarray(X_t))

##input data
input_data = dict()
input_data['PPMI_S'] = PPMI_s
input_data['PPMI_T'] = PPMI_t
input_data['attrb_S'] = X_s
input_data['attrb_T'] = X_t
input_data['attrb_nei_S'] = X_n_s
input_data['attrb_nei_T'] = X_n_t
input_data['label_S'] = Y_s
input_data['label_T'] = Y_t
input_data['source'] = source
input_data['target'] = target
input_data['s_node'] = num_node_s_old

###model config
config = dict()
config['clf_type'] = 'multi-class'
config['dropout'] = 0.6
config['num_epoch'] = 200
config['batch_size'] = 200
config['n_hidden'] = [512, 128]
config['n_emb'] = 128
config['l2_w'] = 1e-3
config['net_pro_w'] = 1e-3
config['open_set_w'] = 0.1
config['emb_filename'] = emb_filename
config['lr_ini'] = 0.01

micro_t, macro_t,known_micro,known_macro,open_acc,known_acc,class_average_acc,best_epoch = train_and_evaluate(input_data, config, random_state)
print('random seed: {:03d} '.format(0))
print("lr: {}  epoch: {}   batchsize: {}   ".format(config['lr_ini'],config['num_epoch'],config['batch_size']))
print("source network: {}   target network: {}".format(source,target))
print("all_class    micro-F1 {:.4f}     macro-F1 {:.4f}".format(micro_t,macro_t))
print("known_class  micro-F1 {:.4f}     known_macro-F1 {:.4f}".format(known_micro,known_macro))
print("known_class  acc {:.4f}".format(known_acc))
print("known_class  average acc {:.4f}".format(class_average_acc))
print("open acc {:.4f}".format(open_acc))