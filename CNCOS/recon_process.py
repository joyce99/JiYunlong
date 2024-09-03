from scipy.sparse import lil_matrix
from scipy.sparse import csc_matrix

from recon_model import *
from sklearn.metrics.pairwise import cosine_similarity




def recon(X_s,Y_s,A_s,X_t,A_t,Y_t):
    num_nodes_S = X_s.shape[0]
    num_nodes_T = X_t.shape[0]
    num_class_S = Y_s.shape[1]
    Y_t_o = np.zeros((Y_t.shape[0],Y_s.shape[1]))
    whole_xs_xt=np.vstack((X_s.toarray(),X_t.toarray()))

    n_input=X_s.shape[1]
    np.random.seed(0)

    adamodel = adatrain(n_input)
    _,pred = adamodel.train(X_s.toarray(), X_t.toarray(), Y_s, Y_t_o,epochs=30, batch_size=100, num_S=num_nodes_S,
                         num_T=num_nodes_T, data3=whole_xs_xt)
    pred_max_row = np.max(pred, axis=1)

    threshold=0.3
    index = np.where(pred_max_row < threshold)[0]
    add_total = np.sum(pred_max_row < threshold)
    print("count of pred below {}: {}".format(threshold,add_total))
    result = lil_matrix(X_t).toarray()[index]

    aug_index = np.random.choice(np.arange(0, num_nodes_S), add_total, False)
    source = X_s.toarray()[aug_index]
    newmodel = Autoencoder(n_input, 128,50)
    X_s_random_array=newmodel.train(result,source, 50, 50, add_total,result)

    X_s_random_array = np.asarray(X_s_random_array)
    X_s_random_array = np.round(X_s_random_array)



    X_s_added = lil_matrix(np.vstack([X_s.toarray(), X_s_random_array]))

    att_sim_s = cosine_similarity(X_s_added.toarray(), X_s_added.toarray())
    for i in range(att_sim_s.shape[0]):
        att_sim_s[i][i]=0
    att_sim_s_known=att_sim_s[num_nodes_S:,:num_nodes_S]
    att_sim_s_unknown=att_sim_s[num_nodes_S:,num_nodes_S:]


    zeros_array = np.zeros((num_nodes_S, 1))
    Y_s = np.hstack((Y_s, zeros_array))
    zeros_array1 = np.zeros((add_total, num_class_S))
    ones_array = np.ones((add_total, 1))
    result_array = np.hstack((zeros_array1, ones_array))
    Y_s_added = np.vstack((Y_s, result_array))


    '''看添边的数量'''
    A_t_arr=A_t.toarray()
    for i in range(A_t_arr.shape[0]):
        for j in range(A_t_arr.shape[0]):
            if A_t_arr[i][j]<0:
                print(True)
    known_edge_num=np.zeros((add_total,))
    unknown_edge_num = np.zeros((add_total,))
    for i in range(add_total):
        row=index[i]
        for j in range(num_nodes_T):
            if j in index:
                unknown_edge_num[i]+=A_t_arr[row][j]
            else:
                known_edge_num[i]+=A_t_arr[row][j]
    unknown_edge_num=unknown_edge_num.astype(int)

    known_edge_num_pro = known_edge_num / (num_nodes_T-add_total)
    known_edge_num=known_edge_num_pro*num_nodes_S
    known_edge_num=known_edge_num.astype(int)

    A_s_lil = lil_matrix(A_s)
    known = 0
    unknown = 0
    add_row1 = np.zeros((add_total, num_nodes_S))

    for i in range(add_total):
        if known_edge_num[i]!=0:
            maxkindex=np.argsort(att_sim_s_known[i])[-known_edge_num[i]:]
            for j in range(num_nodes_S):
                if j in maxkindex:
                    add_row1[i][j]=1
                    known+=1


    add_col = np.transpose(add_row1)
    add_row2 = np.zeros((add_total, add_total))

    for i in range(add_total):
        if unknown_edge_num[i]!=0:
            maxkindex=np.argsort(att_sim_s_unknown[i])[-unknown_edge_num[i]:]
            for j in range(i + 1, add_total):
                if j in maxkindex:
                    add_row2[i][j] = 1
                    unknown+=1
                    add_row2[j, i] = add_row2[i, j]


    A_s_1 = np.vstack((A_s_lil.toarray(), add_row1))
    A_s_2 = np.vstack((add_col, add_row2))
    A_s_arr=np.hstack((A_s_1, A_s_2))
    A_s_3 = lil_matrix(np.hstack((A_s_1, A_s_2)))



    A_s_added = csc_matrix(A_s_3)

    return X_s_added,Y_s_added,A_s_added,add_total
