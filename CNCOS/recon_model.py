import tensorflow as tf
import numpy as np
import utils
from flip_gradient import flip_gradient


class Autoencoder:
    def __init__(self, input_dim, hidden_dim,batchsize):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batchsize=batchsize
        self.build_model()

    def build_model(self):
        # 定义输入占位符
        self.input_data = tf.placeholder(tf.float32,[None,self.input_dim])
        self.mask = tf.placeholder(tf.float32)
        self.encoder = tf.layers.dense(self.input_data, self.hidden_dim, activation=tf.nn.relu)
        self.encoder1 = tf.layers.dense(self.encoder, self.hidden_dim, activation=tf.nn.relu)
        self.target=tf.slice(self.encoder1, [0, 0], [self.batchsize, -1])    #切片，[0,0]表示begin  [int,-1]表示长度
        self.source=tf.slice(self.encoder1, [self.batchsize, 0], [self.batchsize, -1])
        target_mean=tf.reduce_mean(self.target,axis=0)
        target_mean=tf.reshape(target_mean,(1,-1))
        target_mean_exp = tf.tile(target_mean, [self.batchsize, 1])
        source_mean=tf.reduce_mean(self.source,axis=0)
        source_mean=tf.reshape(source_mean,(1,-1))
        source_mean_exp=tf.tile(source_mean,[self.batchsize,1])
        A = tf.Variable(tf.truncated_normal([self.hidden_dim, self.hidden_dim], stddev=1. / tf.sqrt(self.hidden_dim/2.)))
        y1=tf.einsum('ij,jk,ik->i', self.target, A,target_mean_exp)
        y2=tf.einsum('ij,jk,ik->i', self.target, A,source_mean_exp)
        y=tf.concat([y1,y2],axis=0)
        self.decoder1 = tf.layers.dense(self.encoder1, self.hidden_dim, activation=tf.nn.relu)
        self.decoder = tf.layers.dense(self.decoder1, self.input_dim, activation=tf.nn.relu)
        self.reconstruction_loss = tf.reduce_mean(tf.square(self.input_data - self.decoder))
        bce=tf.keras.losses.BinaryCrossentropy()
        self.miloss=bce(self.mask,y)
        self.totalloss=self.reconstruction_loss + 0.01*self.miloss
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.totalloss)

    def train(self, data1,data2, epochs, batch_size,add_total,data3):


        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # 迭代训练
            for epoch in range(epochs):
                S_batches = utils.batch_generator([data1], int(batch_size), shuffle=True)    #目标中的
                T_batches = utils.batch_generator([data2], int(batch_size), shuffle=True)    #源中随便选的

                num_batch = round(int(add_total) / int(batch_size))
                for cBatch in range(num_batch):
                    xs_batch, shuffle_index_s = next(S_batches)
                    xs_batch = xs_batch[0]

                    xt_batch, shuffle_index_t = next(T_batches)
                    xt_batch = xt_batch[0]

                    x_batch = np.vstack((xs_batch, xt_batch))

                    mask = np.vstack(
                        [np.tile([1.], [batch_size, 1]), np.tile([0.], [batch_size, 1])])

                    _, loss,loss1,loss2= sess.run([self.optimizer, self.totalloss,self.reconstruction_loss,self.miloss],
                                       feed_dict={self.input_data: x_batch,self.mask:mask})

                if epoch % 5 == 0:
                    print("Epoch: {}".format(epoch))

            new_attributes = sess.run(self.decoder, feed_dict={self.input_data: data3})
        return new_attributes



    def test(self, data):

        with tf.Session() as sess:

            new_att = sess.run([self.decoder], feed_dict={self.input_data: data})



        return new_att


class adatrain:
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.build_model()

    def build_model(self):
        # 定义输入占位符
        self.input_data = tf.placeholder(tf.float32,[None,self.input_dim])
        self.Ada_lambda = tf.placeholder(tf.float32)
        self.d_label=tf.placeholder(tf.float32)
        self.lr = tf.placeholder(tf.float32)
        self.y_true = tf.placeholder(tf.float32)
        self.mask=tf.placeholder(tf.float32)

        # 定义编码器
        hidden1=tf.layers.dense(self.input_data, self.input_dim//2, activation=tf.nn.relu)
        hidden1 = tf.layers.batch_normalization(hidden1)
        hidden2 = tf.layers.dense(hidden1, self.input_dim // 4, activation=tf.nn.relu)
        hidden2 = tf.layers.batch_normalization(hidden2)

        self.emb = tf.layers.dense(hidden2, 128, activation=tf.nn.relu)
        self.emb = tf.layers.batch_normalization(self.emb)


        pred_logit = tf.layers.dense(self.emb, 5)
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred_logit, labels=self.y_true)
        loss = loss * self.mask
        self.clf_loss = tf.reduce_mean(loss)
        self.pred_prob = tf.nn.softmax(pred_logit)
        # pred_logit = tf.layers.dense(self.emb, 4)
        # loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=pred_logit, labels=self.y_true)
        # loss = loss * self.mask[:, None]  # count loss only based on labeled nodes, each column mutiply by mask
        # self.clf_loss = tf.reduce_sum(loss) / tf.reduce_sum(self.mask)
        # self.pred_prob = tf.sigmoid(pred_logit)


        h_grl = flip_gradient(self.emb, self.Ada_lambda)
        h_dann_1=tf.layers.dense(h_grl, 128, activation=tf.nn.relu)
        h_dann_1 = tf.layers.batch_normalization(h_dann_1)
        h_dann_2=tf.layers.dense(h_dann_1, 128, activation=tf.nn.relu)
        h_dann_2 = tf.layers.batch_normalization(h_dann_2)

        self.d_logit=tf.layers.dense(h_dann_2, 2)

        # 定义损失
        self.domain_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.d_logit, labels=self.d_label))
        self.totalloss=0.1*self.clf_loss+self.domain_loss

        # 定义优化器
        self.optimizer = tf.train.MomentumOptimizer(self.lr,0.9).minimize(self.totalloss)

    def train(self, data1,data2,Y_s,Y_t_o,epochs, batch_size,num_S,num_T,data3):

        lr_ini=0.01


        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # 迭代训练
            for epoch in range(epochs):
                S_batches = utils.batch_generator([data1,Y_s], int(batch_size / 2), shuffle=True)
                T_batches = utils.batch_generator([data2,Y_t_o], int(batch_size / 2), shuffle=True)

                num_batch = round(max(num_S / (batch_size / 2), num_T / (batch_size / 2)))

                # Adaptation param and learning rate schedule as described in the DANN paper
                p = float(epoch) / (epochs)
                lr = lr_ini / (1. + 10 * p) ** 0.75
                grl_lambda = 2. / (1. + np.exp(-10. * p)) - 1  # gradually change from 0 to 1

                ##in each epoch, train all the mini batches
                for cBatch in range(num_batch):
                    ### each batch, half nodes from source network, and half nodes from target network
                    xs_ys_batch, shuffle_index_s = next(S_batches)
                    xs_batch = xs_ys_batch[0]
                    ys_batch = xs_ys_batch[1]

                    xt_yt_batch, shuffle_index_t = next(T_batches)
                    xt_batch = xt_yt_batch[0]
                    yt_batch = xt_yt_batch[1]

                    x_batch = np.vstack((xs_batch,xt_batch))
                    y_true=np.vstack((ys_batch,yt_batch))


                    domain_label = np.vstack(
                        [np.tile([1., 0.], [batch_size // 2, 1]), np.tile([0., 1.], [batch_size // 2, 1])])

                    mask_L = np.array(np.sum(y_true, axis=1) > 0, dtype=np.float)

                    _,clfloss,domainloss = sess.run([self.optimizer,self.clf_loss,self.domain_loss],
                                       feed_dict={self.input_data: x_batch,self.Ada_lambda:grl_lambda,self.d_label:domain_label,self.lr:lr,self.y_true:y_true,self.mask:mask_L})

                pred_prob_xs_xt = sess.run(self.pred_prob,
                                           feed_dict={self.input_data: data3})
                pred_prob_xs = pred_prob_xs_xt[0:num_S, :]

                F1_s = utils.f1_scores(pred_prob_xs, Y_s)
                print("Epoch: {}, Source micro-F1:{}  macro-F1: {}".format(epoch,F1_s[0], F1_s[1]))


            emb = sess.run(self.emb, feed_dict={self.input_data: data3})
            pred=sess.run(self.pred_prob,feed_dict={self.input_data:data3})
            pred=pred[-num_T:,:]
        return emb,pred





