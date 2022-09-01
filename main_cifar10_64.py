from my_generator import Vgg19
from scipy.spatial.distance import pdist ,squareform
from scipy.stats import norm
#import pdb
import random
import tqdm
import prettytensor as pt
import scipy
import h5py
import scipy.io as sio
import tensorflow as tf
import numpy as np
from ops import *
from utils import *
import pdb

import os, sys
sys.path.append(os.getcwd())
os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
import batch_data.mxnet_image_data as dataset
import numpy as np
import sklearn.datasets
import time
import functools
import locale
locale.setlocale(locale.LC_ALL, '')
import os
def label_network(label, label_dim, code_dim):
    hidden_dim = int(np.ceil((label_dim + code_dim)/2))
    initial_value = tf.truncated_normal([label_dim, hidden_dim],0.0, 0.5)
    weights1 = tf.Variable(initial_value, name = 'label_network_1')
    initial_value = tf.truncated_normal([hidden_dim], 0.0, 0.5)
    biases1 = tf.Variable(initial_value, name = 'label_network_1_b')
    label_fc1 = tf.nn.bias_add(tf.matmul(label, weights1), biases1)

    initial_value = tf.truncated_normal([hidden_dim, code_dim],0.0, 0.5)
    weights2 = tf.Variable(initial_value, name = 'label_network_2')
    initial_value = tf.truncated_normal([code_dim], 0.0, 0.5)
    biases2 = tf.Variable(initial_value, name = 'label_network_2_b')
    label_fc2 = tf.nn.bias_add(tf.matmul(label_fc1, weights2), biases2)
    label_output = tf.nn.tanh(label_fc2)
    return label_output
    
def conv_out_size_same(size, stride):
  return int(math.ceil(float(size) / float(stride)))

def calc_map_k(qB, rB, query_L, retrieval_L, k):
    num_query = query_L.shape[0]
    map = 0
    for iter in xrange(num_query):
        gnd2 = (np.dot(query_L[iter, :], retrieval_L.transpose()) > 0).astype(np.float32)
        hamm = calc_hammingDist(qB[iter, :], rB)
        aaa = np.arange(0,retrieval_L.shape[0])
        ind = np.lexsort((aaa,hamm))
        gnd = gnd2[ind[0:k]]
        tsum = np.sum(gnd)
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)

        tindex = np.asarray(np.where(gnd == 1)) + 1.0
        map = map + np.mean(count / (tindex))
    map = map / num_query
    return map
def calc_map(qB, rB, query_L, retrieval_L):
    num_query = query_L.shape[0]
    map = 0
    for iter in xrange(num_query):
        gnd2 = (np.dot(query_L[iter, :], retrieval_L.transpose()) > 0).astype(np.float32)
        hamm = calc_hammingDist(qB[iter, :], rB)
        aaa = np.arange(0,retrieval_L.shape[0])
        ind = np.lexsort((aaa,hamm))
        gnd = gnd2[ind[:]]
        tsum = np.sum(gnd)
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)

        tindex = np.asarray(np.where(gnd == 1)) + 1.0
        map = map + np.mean(count / (tindex))
    map = map / num_query
    return map
def calc_hammingDist(B1, B2):
    B1 = B1*1
    B2 = B2*1
    #pdb.set_trace()
    ind = B1<0.5
    B1[ind] = -1
    ind = B2<0.5
    B2[ind] = -1
    q = B2.shape[1]
    distH = 0.5 * (q - np.dot(B1,B2.transpose()))
    return distH
def loss(z_x_meanx1,label_output, ss_ ):
    # for pair_loss ss_ be 1 or -1; for cross_entropy_loss ss_ be 1 or 0
    pair_loss=tf.reduce_mean(tf.multiply(tf.abs(ss_),(tf.square(tf.multiply(1.0/hidden_size,tf.matmul(z_x_meanx1, tf.transpose(label_output)))- ss_))))

    return pair_loss
    #return cross_entropy_loss
def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

def inference(x224):
    #z_p = tf.random_normal((batch_size, hidden_size), 0, 1)
    #eps = tf.random_normal((batch_size, hidden_size), 0, 1)  # normal dist for VAE
    with pt.defaults_scope(activation_fn=tf.nn.elu,
                           batch_normalize=True,
                           learned_moments_update_rate=0.0003,
                           variance_epsilon=0.001,
                           scale_after_normalization=True):
        with tf.variable_scope("enc"):
                vgg_net = Vgg19('./vgg16.npy', codelen=hidden_size)
                vgg_net.build(x224, train_model)
                z_x = vgg_net.fc9
                fc7_features = vgg_net.relu7
        return z_x, fc7_features
batch_size = 24 
hidden_size = 64
k = 50
config = {
    
    'img_tr': "cifar10/img_train.txt", 
    'lab_tr': "cifar10/label_train.txt",
    'img_te': "cifar10/img_test.txt",
    'lab_te': "cifar10/label_test.txt",
    'img_db': "cifar10/img_database.txt", 
    'lab_db': "cifar10/label_database.txt",
    'n_train': 5000,
    'n_test': 10000,
    'n_db': 50000,
    'n_label': 10
}

n_train = config['n_train']
n_test = config['n_test']
n_db = config['n_db']
n_label = config['n_label']
train_size = n_train
test_size = n_test
db_size = n_db
label_dim = n_label
train_data = dataset.import_train(config)
train_labels = np.zeros([n_train, n_label])
train_features = np.zeros([ n_train, 4096])
test_data = dataset.import_test(config)
test_labels = np.zeros([n_test, n_label])
db_data = dataset.import_db(config)
db_labels = np.zeros([n_db, n_label])
all_input224 = tf.placeholder(tf.float32, [None, 224 ,224,3])
all_input64 = tf.placeholder(tf.float32, [None, 64, 64, 3])
label_input = tf.placeholder(tf.float32, [None, label_dim])
label_output = label_network(label_input,n_label, hidden_size)
train_model = tf.placeholder(tf.bool)
s_s = tf.placeholder(tf.float32, [batch_size, batch_size])
with tf.device('/gpu:0'):
    with tf.name_scope('Tower_0') as scope:
        z_x, fc7_features = inference(all_input224)
        pair_loss = loss(z_x,label_output, s_s)
        params = tf.trainable_variables()
        print params
        #pdb.set_trace()
        E_params = [i for i in params if 'enc' in i.name or 'label' in i.name]
        lr_E = tf.placeholder(tf.float32, shape=[])
        opt_E = tf.train.AdamOptimizer(lr_E, epsilon=1.0)            
        grads_e = opt_E.compute_gradients(pair_loss, var_list=E_params)#with KL_loss,you can discard it.
#with graph.as_default():
global_step = tf.get_variable(
    'global_step', [],
    initializer=tf.constant_initializer(0), trainable=False)
train_E = opt_E.apply_gradients(grads_e, global_step=global_step)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 1.0)
session = tf.Session(config = tf.ConfigProto(gpu_options = gpu_options))
cur_epoch = 0
num_epochs =  100
e_learning_rate = 1e-3
g_learning_rate = 1e-3
d_learning_rate = 1e-3
globa_beta_indx = 0
num_examples = n_train
total_batch = int(np.floor(num_examples / batch_size))   
session.run(tf.initialize_all_variables())
epoch = 0  
pre_epochs =  100
train_batch = int(np.ceil(1.0*n_train/batch_size))
while epoch < pre_epochs:
    iter_ = data_iterator()
    index_range = np.arange(n_train)
    np.random.shuffle(index_range)
    for i in  range(total_batch):
        if (i+1)*batch_size < n_train:
            index = index_range[range(i*batch_size,(i+1)*batch_size)]
        else:
            index = index_range[range(n_train - batch_size, n_train)]
        cur_epoch += 1.0
        e_current_lr = e_learning_rate*1.0
        next_batches224,batch_label = train_data.img_data(index)
        next_batches224 = np.array(next_batches224)
        next_batches64 = resize_32(next_batches224)
        ss_ = (np.matmul(batch_label, np.transpose(batch_label))>0)*2 - 1
        _, PP_err= session.run(
            [
             train_E, pair_loss
             ],
            {
                lr_E: e_current_lr,
                all_input224: next_batches224,
                label_input: batch_label,
                s_s: ss_,
                train_model: True
            }
            )
        print 'epoch:{}, batch: {},PP_err:{}'.format(epoch,i, PP_err)
    epoch = epoch + 1

    if (epoch+1) % 1 ==0 :
        test_codes = np.zeros([n_test,hidden_size])
        test_labels = np.zeros([n_test,n_label])
        dataset_codes = np.zeros([n_db,hidden_size])
        dataset_labels = np.zeros([n_db, n_label])
        
        test_batch = int(np.ceil(1.0*test_size/batch_size))
        dataset_batch =int(np.ceil(1.0*db_size/batch_size))


        for i in range(test_batch):
            if (i+1)*batch_size < n_test:
                index = range(i*batch_size,(i+1)*batch_size)
            else:
                index = range(i*batch_size, n_test)
            next_batches224, batch_label = test_data.img_data(index)
            
            next_batches224 = np.array(next_batches224)
            next_batches64 = resize_32(next_batches224)
            test_softcode = session.run(z_x, feed_dict = {all_input224: next_batches224, train_model: False})
            test_codes[index, :] = test_softcode
            test_labels[index,:] = batch_label

        for i in range(dataset_batch):
            if (i+1)*batch_size < n_db:
                index = range(i*batch_size,(i+1)*batch_size)
            else:
                index = range(i*batch_size, n_db)
            next_batches224, batch_label = db_data.img_data(index)
            next_batches224 = np.array(next_batches224)
            dataset_softcode = session.run(label_output, feed_dict = {label_input: batch_label})
            dataset_codes[index, :] = dataset_softcode
            dataset_labels[index, :] = batch_label

        dataset_codes = (dataset_codes>0)*1
        test_codes = (test_codes>0)*1
        dataset_L = dataset_labels 
        test_L = test_labels  #cifar-10 data
        dict_ = {'dataset_codes':dataset_codes, 'test_codes': test_codes, 'dataset_L': dataset_L, 'test_L': test_L}
        map_1000 = calc_map_k(test_codes, dataset_codes, test_L, dataset_L, 1000)
        map_ = calc_map(test_codes, dataset_codes, test_L, dataset_L)
        print 'pre: epoch:{}, map_1000:{}, map:{}'.format(epoch, map_1000, map_)
        np.save('./result/cifar10/64bit/pre1'+str(epoch) +'.npy', dict_)
