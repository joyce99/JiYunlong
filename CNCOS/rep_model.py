import tensorflow as tf


import utils
from flip_gradient import flip_gradient


class ACDNE(object):
    def __init__(self, n_input, n_hidden, n_emb, num_class_s, clf_type, l2_w, net_pro_w, open_set_w, batch_size):

        self.X = tf.sparse_placeholder(dtype=tf.float32)
        self.X_nei = tf.sparse_placeholder(dtype=tf.float32)
        self.y_true = tf.placeholder(dtype=tf.float32)
        self.d_label = tf.placeholder(dtype=tf.float32)
        self.Ada_lambda = tf.placeholder(dtype=tf.float32)
        self.dropout = tf.placeholder(dtype=tf.float32)
        self.A_s = tf.sparse_placeholder(dtype=tf.float32)
        self.A_t = tf.sparse_placeholder(dtype=tf.float32)
        self.A_sim_s = tf.sparse_placeholder(dtype=tf.float32)
        self.A_sim_t = tf.sparse_placeholder(dtype=tf.float32)
        self.mask = tf.placeholder(dtype=tf.float32)
        self.learning_rate = tf.placeholder(dtype=tf.float32)

        with tf.name_scope('Network_Embedding'):
            ##feature exactor 1
            h1_self = utils.fc_layer(self.X, n_input, n_hidden[0], layer_name='hidden1_self', input_type='sparse',
                                     drop=self.dropout)
            h2_self = utils.fc_layer(h1_self, n_hidden[0], n_hidden[1], layer_name='hidden2_self')

            ##feature exactor 2
            h1_nei = utils.fc_layer(self.X_nei, n_input, n_hidden[0], layer_name='hidden1_nei', input_type='sparse',
                                    drop=self.dropout)
            h2_nei = utils.fc_layer(h1_nei, n_hidden[0], n_hidden[1], layer_name='hidden2_nei')

            ##concatenation layer, final embedding vector representation
            self.emb = utils.fc_layer(tf.concat([h2_self, h2_nei], 1), n_hidden[-1] * 2, n_emb, layer_name='concat')

            emb_s = tf.slice(self.emb, [0, 0], [int(batch_size / 2), -1])
            emb_t = tf.slice(self.emb, [int(batch_size / 2), 0], [int(batch_size / 2), -1])
            r_s = tf.reduce_sum(emb_s * emb_s, 1)
            r_s = tf.reshape(r_s, [-1, 1])
            Dis_s = r_s - 2 * tf.matmul(emb_s, tf.transpose(emb_s)) + tf.transpose(r_s)
            net_pro_loss_s = tf.reduce_mean(tf.sparse.reduce_sum(self.A_s.__mul__(Dis_s), axis=1))
            r_t = tf.reduce_sum(emb_t * emb_t, 1)
            r_t = tf.reshape(r_t, [-1, 1])
            Dis_t = r_t - 2 * tf.matmul(emb_t, tf.transpose(emb_t)) + tf.transpose(r_t)
            net_pro_loss_t = tf.reduce_mean(tf.sparse.reduce_sum(self.A_t.__mul__(Dis_t), axis=1))
            self.net_pro_loss = net_pro_w * (net_pro_loss_s + net_pro_loss_t)



        with tf.name_scope('open-set'):

            label_s = tf.slice(self.y_true, [0, 0], [int(batch_size / 2), num_class_s])
            counts = tf.reduce_sum(label_s, axis=0)
            counts = tf.reshape(counts, [-1, 1])
            sums = tf.matmul(tf.transpose(label_s), emb_s)
            averages = tf.divide(sums, counts)

            if clf_type == 'multi-class':
                indice1 = tf.where(tf.equal(label_s, 0))
                indices1 = tf.gather(indice1, indices=1, axis=1)
                otherlabel_emb = tf.gather(averages, indices1, axis=0)
                tiled_tensor = tf.tile(emb_s[:, None, :], [1, num_class_s - 1, 1])
                output_tensor = tf.reshape(tiled_tensor, [tf.shape(emb_s)[0] * (num_class_s - 1), tf.shape(emb_s)[1]])
                output_tensor_2 = output_tensor * output_tensor
                otherlabel_emb_2 = otherlabel_emb * otherlabel_emb
                Dis_Denominator = output_tensor_2 - 2 * output_tensor * otherlabel_emb + otherlabel_emb_2
                indice2 = tf.where(tf.equal(label_s, 1))
                indices2 = tf.gather(indice2, indices=1, axis=1)
                thislabel_emb = tf.gather(averages, indices2, axis=0)
                emb_s_2 = emb_s * emb_s
                thislabel_emb_2 = thislabel_emb * thislabel_emb
                Dis_Numerator = emb_s_2 - 2 * emb_s * thislabel_emb + thislabel_emb_2
                loss_Denominator = tf.reduce_sum(Dis_Denominator)
                loss_Numerator = tf.reduce_sum(Dis_Numerator)
                self.openset_loss = open_set_w * (loss_Numerator / loss_Denominator)

            elif clf_type == 'multi-label':
                indice1 = tf.where(tf.equal(label_s, 0))
                indices1 = tf.gather(indice1, indices=1, axis=1)
                otherlabel_emb = tf.gather(averages, indices1, axis=0)
                zero_count = tf.reduce_sum(tf.cast(tf.equal(label_s, 0), tf.int32), axis=1)

                output_list = []
                for i in range(int(batch_size / 2)):
                    row = emb_s[i, :]
                    repeat_num = zero_count[i]
                    tiled_row = tf.tile(tf.expand_dims(row, axis=0), multiples=[repeat_num, 1])
                    output_list.append(tiled_row)

                output_tensor = tf.reshape(tf.concat(output_list, axis=0), shape=[-1, n_emb])
                output_tensor_2 = output_tensor * output_tensor
                otherlabel_emb_2 = otherlabel_emb * otherlabel_emb
                Dis_Denominator = output_tensor_2 - 2 * output_tensor * otherlabel_emb + otherlabel_emb_2
                indice2 = tf.where(tf.equal(label_s, 1))
                indices2 = tf.gather(indice2, indices=1, axis=1)
                thislabel_emb = tf.gather(averages, indices2, axis=0)
                zero_count2 = tf.reduce_sum(tf.cast(tf.equal(label_s, 1), tf.int32), axis=1)

                output_list1 = []
                for i in range(int(batch_size / 2)):
                    row = emb_s[i, :]
                    repeat_num1 = zero_count2[i]
                    tiled_row = tf.tile(tf.expand_dims(row, axis=0), multiples=[repeat_num1, 1])
                    output_list1.append(tiled_row)

                emb_s_output = tf.reshape(tf.concat(output_list1, axis=0), shape=[-1, n_emb])
                emb_s_2 = emb_s_output * emb_s_output
                thislabel_emb_2 = thislabel_emb * thislabel_emb
                Dis_Numerator = emb_s_2 - 2 * emb_s_output * thislabel_emb + thislabel_emb_2
                loss_Denominator = tf.reduce_sum(Dis_Denominator)
                loss_Numerator = tf.reduce_sum(Dis_Numerator)

                self.openset_loss = open_set_w * (loss_Numerator / loss_Denominator)

        with tf.name_scope('Node_Classifier'):
            ##node classification
            W_clf = tf.Variable(tf.truncated_normal([n_emb, num_class_s], stddev=1. / tf.sqrt(n_emb / 2.)),
                                name='clf_weight')
            b_clf = tf.Variable(tf.constant(0.1, shape=[num_class_s]), name='clf_bias')
            pred_logit = tf.matmul(self.emb, W_clf) + b_clf

            if clf_type == 'multi-class':
                loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred_logit, labels=self.y_true)
                loss = loss * self.mask  # count loss only based on labeled nodes
                self.clf_loss = tf.reduce_sum(loss) / tf.reduce_sum(self.mask)
                self.pred_prob = tf.nn.softmax(pred_logit)

            elif clf_type == 'multi-label':
                loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=pred_logit, labels=self.y_true)
                loss = loss * self.mask[:, None]  # count loss only based on labeled nodes, each column mutiply by mask
                self.clf_loss = tf.reduce_sum(loss) / tf.reduce_sum(self.mask)
                self.pred_prob = tf.sigmoid(pred_logit)

        with tf.name_scope('Domain_Discriminator'):
            h_grl = flip_gradient(self.emb, self.Ada_lambda)
            h_dann_1 = utils.fc_layer(h_grl, n_emb, 128, layer_name='dann_fc_1')
            h_dann_2 = utils.fc_layer(h_dann_1, 128, 128, layer_name='dann_fc_2')
            W_domain = tf.Variable(tf.truncated_normal([128, 2], stddev=1. / tf.sqrt(128 / 2.)), name='dann_weight')
            b_domain = tf.Variable(tf.constant(0.1, shape=[2]), name='dann_bias')
            d_logit = tf.matmul(h_dann_2, W_domain) + b_domain
            self.d_softmax = tf.nn.softmax(d_logit)
            self.domain_loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(logits=d_logit, labels=self.d_label))

        all_variables = tf.trainable_variables()
        self.l2_loss = l2_w * tf.add_n([tf.nn.l2_loss(v) for v in all_variables if 'bias' not in v.name])

        self.total_loss = self.net_pro_loss + self.clf_loss + self.domain_loss + self.l2_loss + self.openset_loss

        self.train_op = tf.train.MomentumOptimizer(self.learning_rate, 0.9).minimize(self.total_loss)


