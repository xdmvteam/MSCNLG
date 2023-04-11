import tensorflow as tf
import numpy as np
from tensorflow.contrib import layers
import scipy.io as sio
from scipy.sparse.linalg import svds
from sklearn import cluster
from sklearn.preprocessing import normalize, minmax_scale
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
from munkres import Munkres
from metrics import acc, randIndex, f_score
from sklearn.decomposition import PCA


class AE_full(object):
    def __init__(self, L1_all, L2_all, sample_num, enc_dim_list, dec_dim_list, num_views=2, para_1=1.0, para_2=1.0,
                 para_3=1.0, learning_rate=1e-3, reg=None, model_path=None, restore_path=None,
                 logs_path='./Net_Models_logs/YaleFace_logs_multiview'):
        # "alpha" used in paper is "para_1" here.
        # "beta" used in paper is "para_2" for "MSCNLG_1st" ("para_2" and "para_3" for "MSCNLG") here

        self.sample_num = sample_num
        self.enc_dim_list = enc_dim_list  # matrix
        self.dec_dim_list = dec_dim_list  # matrix
        self.reg = reg
        self.model_path = model_path
        self.restore_path = restore_path
        self.iter = 0
        self.num_views = num_views
        self.learning_rate = learning_rate
        self.L1_all = L1_all
        self.L2_all = L2_all
        weights = self._initialize_weights()

        self.x = {}
        for i in range(0, self.num_views):
            modality = str(i)
            self.x[modality] = tf.placeholder(tf.float32, [None, enc_dim_list[i][0]])

        latents = self.encoder(self.x, weights, num_views)
        z = latents
        self.z = z
        coef = weights['coef']
        self.coef = coef

        z_r = {}
        for i in range(0, num_views):
            modality = str(i)
            z_r[modality] = tf.matmul(self.coef, self.z[modality])

        self.x_r = self.decoder(z_r, weights, num_views)
        self.saver = tf.train.Saver()

        # reconstruction loss
        self.reconst_loss = 0.5*tf.reduce_sum(tf.pow(tf.subtract(self.x_r['0'], self.x['0']), 2.0))
        for i in range(1, num_views):
            modality = str(i)
            self.reconst_loss = self.reconst_loss + 0.5*tf.reduce_sum(tf.pow(tf.subtract(self.x_r[modality], self.x[modality]), 2.0))
        tf.summary.scalar("l2_loss", self.reconst_loss)

        # regularizer loss
        tmp_g1 = tf.matmul(tf.transpose(self.coef), tf.cast(L1_all, tf.float32))
        self.smooth_loss_1 = tf.trace(tf.matmul(tmp_g1, self.coef))

        tf.summary.scalar("smooth_loss_1", self.smooth_loss_1)

        tmp_g2 = tf.matmul(tf.transpose(self.coef), tf.cast(L2_all, tf.float32))
        self.smooth_loss_2 = tf.trace(tf.matmul(tmp_g2, self.coef))

        tf.summary.scalar("smooth_loss_2", self.smooth_loss_2)

        # selfexpress_loss
        self.selfexpress_loss = 0.5*tf.reduce_sum(tf.pow(tf.subtract(z_r['0'], z['0']), 2.0))
        for i in range(1, num_views):
            modality = str(i)
            self.selfexpress_loss = self.selfexpress_loss + 0.5*tf.reduce_sum(tf.pow(tf.subtract(z_r[modality], z[modality]), 2.0))
        tf.summary.scalar("selfexpress_loss", self.selfexpress_loss) 

        self.loss = self.reconst_loss + para_1*self.selfexpress_loss + para_2*self.smooth_loss_1 + para_3*self.smooth_loss_2

        self.merged_summary_op = tf.summary.merge_all()

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)                     
        
        self.init = tf.global_variables_initializer()
        self.sess = tf.InteractiveSession()
        self.sess.run(self.init)
        self.saver = tf.train.Saver([v for v in tf.trainable_variables() if not (v.name.startswith("coef"))])
        self.summary_weiter = tf.summary.FileWriter(logs_path,graph=tf.get_default_graph())

        t_vars = tf.trainable_variables()

    def _initialize_weights(self):
        all_weights = dict()
        
        for i in range(0, self.num_views):
            modality = str(i)
            with tf.variable_scope(modality):
                # encoder layer 1
                all_weights[modality+'_enc_w0'] = tf.get_variable(modality+"_enc_w0", shape=[self.enc_dim_list[i][0], self.enc_dim_list[i][1]],
                    initializer=layers.xavier_initializer(),regularizer=self.reg)
                all_weights[modality+'_enc_b0'] = tf.Variable(tf.zeros([self.enc_dim_list[i][1]], dtype=tf.float32))
                # encoder layer 2
                all_weights[modality+'_enc_w1'] = tf.get_variable(modality+"_enc_w1", shape=[self.enc_dim_list[i][1], self.enc_dim_list[i][2]],
                    initializer=layers.xavier_initializer(),regularizer=self.reg)
                all_weights[modality+'_enc_b1'] = tf.Variable(tf.zeros([self.enc_dim_list[i][2]], dtype=tf.float32))

                
                # decoder layer 1
                all_weights[modality+'_dec_w0'] = tf.get_variable(modality+"_dec_w0", shape=[self.enc_dim_list[i][2], self.dec_dim_list[i][0]],
                    initializer=layers.xavier_initializer(),regularizer=self.reg)
                all_weights[modality+'_dec_b0'] = tf.Variable(tf.zeros([self.dec_dim_list[i][0]], dtype=tf.float32))
                # decoder layer 2
                all_weights[modality+'_dec_w1'] = tf.get_variable(modality+"_dec_w1", shape=[self.dec_dim_list[i][0], self.dec_dim_list[i][1]],
                    initializer=layers.xavier_initializer(),regularizer=self.reg)
                all_weights[modality+'_dec_b1'] = tf.Variable(tf.zeros([self.dec_dim_list[i][1]], dtype=tf.float32))

        all_weights['coef'] = tf.Variable(1.0e-4*tf.ones([self.sample_num, self.sample_num], tf.float32), name='coef')
        
        return all_weights

    def encoder(self, x, weights, num_views):
        # layer 1
        latents = {}
        for i in range(0, num_views):
            modality = str(i)
            layers1 = tf.add(tf.matmul(x[modality], weights[modality+'_enc_w0']), weights[modality+'_enc_b0'])
            layers1 = tf.nn.relu(layers1)
            # layer 2
            layers2 = tf.add(tf.matmul(layers1, weights[modality+'_enc_w1']), weights[modality+'_enc_b1'])
            layers2 = tf.nn.relu(layers2)
            latents[modality] = layers2

        return latents

    def decoder(self, z, weights, num_views):
        recons = {}
        for i in range(0, num_views):
            modality = str(i)    
            # layer 1
            layers1 = tf.add(tf.matmul(z[modality], weights[modality+'_dec_w0']), weights[modality+'_dec_b0'])
            layers1 = tf.nn.relu(layers1)
            # layer 2
            layers2 = tf.add(tf.matmul(layers1, weights[modality+'_dec_w1']), weights[modality+'_dec_b1'])
            layers2 = tf.nn.relu(layers2)
            recons[modality] = layers2

        return recons

    def initlization(self):
        self.sess.run(self.init)

    def restore(self):
        self.saver.restore(self.sess, self.restore_path)
        print ("Model restored from the pretrained model!")

    def partial_fit(self, X, lr):
        feed_dict = {}
        for i in range(0, len(X)):
            feed_dict[self.x[str(i)]] = X[str(i)]
        loss, summary, _, coef, lat_rep_1, lat_rep_2 = self.sess.run((self.loss, self.merged_summary_op, self.optimizer, self.coef, self.z['0'], self.z['1']), feed_dict = feed_dict)
        self.summary_weiter.add_summary(summary, self.iter)
        self.iter = self.iter + 1

        return loss, coef, lat_rep_1, lat_rep_2

    def save_model(self):
        save_path = self.saver.save(self.sess, self.model_path)
        print ("model saved in file: %s" % save_path)


def best_map(L1,L2):
    # L1 should be the groundtruth labels and L2 should be the clustering labels we got
    Label1 = np.unique(L1)
    nClass1 = len(Label1)
    Label2 = np.unique(L2)
    nClass2 = len(Label2)
    nClass = np.maximum(nClass1,nClass2)
    G = np.zeros((nClass,nClass))
    for i in range(nClass1):
        ind_cla1 = L1 == Label1[i]
        ind_cla1 = ind_cla1.astype(float)
        for j in range(nClass2):
            ind_cla2 = L2 == Label2[j]
            ind_cla2 = ind_cla2.astype(float)
            G[i,j] = np.sum(ind_cla2 * ind_cla1)
    m = Munkres()
    index = m.compute(-G.T)
    index = np.array(index)
    c = index[:,1]
    newL2 = np.zeros(L2.shape)
    for i in range(nClass2):
        newL2[L2 == Label2[i]] = Label1[c[i]]
    return newL2   


def thrC(C,ro):
    if ro < 1:
        N = C.shape[1]
        Cp = np.zeros((N,N))
        S = np.abs(np.sort(-np.abs(C),axis=0))
        Ind = np.argsort(-np.abs(C),axis=0)
        for i in range(N):
            cL1 = np.sum(S[:,i]).astype(float)
            stop = False
            csum = 0
            t = 0
            while(stop == False):
                csum = csum + S[t,i]
                if csum > ro*cL1:
                    stop = True
                    Cp[Ind[0:t+1,i],i] = C[Ind[0:t+1,i],i]
                t = t + 1
    else:
        Cp = C

    return Cp


def build_aff(C):
    N = C.shape[0]
    Cabs = np.abs(C)
    ind = np.argsort(-Cabs,0)
    for i in range(N):
        Cabs[:,i]= Cabs[:,i] / (Cabs[ind[0,i],i] + 1e-6)
    Cksym = Cabs + Cabs.T
    return Cksym


def post_proC(C, K, d, alpha):
    # C: coefficient matrix, K: number of clusters, d: dimension of each subspace
    C = 0.5*(C + C.T)
    r = d*K + 1
    U, S, _ = svds(C,r,v0 = np.ones(C.shape[0]))
    U = U[:,::-1]    
    S = np.sqrt(S[::-1])
    S = np.diag(S)    
    U = U.dot(S)    
    U = normalize(U, norm='l2', axis = 1)       
    Z = U.dot(U.T)
    Z = Z * (Z>0)    
    L = np.abs(Z ** alpha) 
    L = L/L.max()   
    L = 0.5 * (L + L.T)    
    spectral = cluster.SpectralClustering(n_clusters=K, eigen_solver='arpack', affinity='precomputed',assign_labels='discretize')
    spectral.fit(L)
    grp = spectral.fit_predict(L) + 1
    return grp, L


def evaluation(gt_s, s):
    c_x = best_map(gt_s,s)
    err_x = np.sum(gt_s[:] != c_x[:])
    nmi = normalized_mutual_info_score(gt_s[:], c_x[:])
    # ari = adjusted_rand_score(gt_s[:], c_x[:])
    missrate = err_x.astype(float) / (gt_s.shape[0])
    acc = 1 - missrate
    ri = randIndex(gt_s[:], c_x[:])
    fscore = f_score(gt_s[:], c_x[:])
    return nmi, acc, fscore, ri, c_x


def build_laplacian(C):
    C = 0.5 * (np.abs(C) + np.abs(C.T))
    W = np.sum(C,axis=0)         
    W = np.diag(1.0/W)
    L = W.dot(C)    
    return L


def train_AE(x, gt, AE, num_class, sample_num, num_views):
    alpha = max(0.4 - (num_class-1)/10 * 0.1, 0.1)
    acc_= []
    nmi_= []
    fscore_= []
    ri_= []
    pred_=[]
    true_=[]

    for i in range(1):
        AE.initlization()
        #AE.restore()

        max_step = 350
        lr = 1.0e-3
        epoch = 0

        acc_curr_iter = []
        nmi_curr_iter = []
        fscore_curr_iter = []
        ri_curr_iter = []
        cost_curr_iter = []

        while epoch < max_step:
            epoch = epoch + 1
            cost, coef, lat_rep_1, lat_rep_2 = AE.partial_fit(x, lr)
            coef = thrC(coef, alpha)
            y_x, _ = post_proC(coef, num_class, 3, 2)
            nmi, acc, fscore, ri, pred1 = evaluation(gt, y_x)

            acc_x = acc
            acc_curr_iter.append(acc_x)
            nmi_curr_iter.append(nmi)
            fscore_curr_iter.append(fscore)
            ri_curr_iter.append(ri)
            cost_curr_iter.append(cost)

        acc_.append(max(acc_curr_iter))
        nmi_.append(max(nmi_curr_iter))
        fscore_.append(max(fscore_curr_iter))
        ri_.append(max(ri_curr_iter))
        pred_.append(pred1)
        true_.append(gt)

    pred_=np.vstack(np.asarray(pred_))
    true_=np.vstack(np.asarray(true_))
    nmi_mean = np.mean(nmi_)
    nmi_std = np.std(nmi_)
    acc_mean = np.mean(acc_)
    acc_std = np.std(acc_)
    # acc_median = np.median(acc_)
    fscore_mean = np.mean(fscore_)
    fscore_std = np.std(fscore_)
    RI_mean = np.mean(ri_)
    RI_std = np.std(ri_)

    print("######################################################################")
    print("Experiment conducted on the YaleFace dataset")
    print("######################################################################")
    print("%d subjects:" % num_class)
    print("NMI: %.4f " % nmi_mean)
    print("ACC: %.4f " % acc_mean)
    print("F-score: %.4f " % fscore_mean)
    print("RI: %.4f " % RI_mean)

    return nmi_mean, acc_mean, fscore_mean, RI_mean, pred_, true_


def load_data(file_name):
    dataset = sio.loadmat(file_name)
    x1, x2, gt = dataset['x1_train'], dataset['x2_train'], dataset['gt_train']
    gt = gt.flatten()
        
    return x1, x2, gt


if __name__ == '__main__':

    num_views = 2
    sample_num = 480

    x0, x1, gt = load_data('F:/dataset/CUB.mat')
    pca = PCA(n_components=30)
    x1 = pca.fit_transform(x1)
    x0 = minmax_scale(x0)
    x1 = minmax_scale(x1)
    x = {}
    x['0'] = x0
    x['1'] = x1


    s_tmp = cluster.SpectralClustering(n_clusters=10, n_neighbors=20, eigen_solver="arpack")

    W1 = {}

    s_tmp.fit(x['0'])
    W1['0'] = s_tmp.affinity_matrix_
    s_tmp.fit(x['1'])
    W1['1'] = s_tmp.affinity_matrix_

    L1 = {}
    L1_all = np.ones([sample_num, sample_num])
    for i in range(0, num_views):
        L1[str(i)] = build_laplacian(W1[str(i)])
        L1_all = L1_all*L1[str(i)]

    W2 = {}
    for i in range(0, num_views):
        s_tmp.fit(W1[str(i)])
        W2[str(i)] = s_tmp.affinity_matrix_

    L2 = {}
    L2_all = np.ones([sample_num, sample_num])
    for i in range(0, num_views):
        L2[str(i)] = build_laplacian(W2[str(i)])
        L1_all = L1_all * L2[str(i)]

    enc_dim_list = [[1024, 1024, 128], [30, 256, 128]]
    dec_dim_list = [[1024, 1024], [128, 30]]

    para_1 = 0.1
    para_2 = 100
    para_3 = 100
    num_class = 10
    model_path = './Net_Models_logs/Model_multiview/YaleFace_model.ckpt'
    restore_path = './Net_Models_logs/Model_multiview/YaleFace_model.ckpt'

    tf.reset_default_graph()

    AE = AE_full(L1_all=L1_all, L2_all=L2_all, sample_num=sample_num, enc_dim_list=enc_dim_list,
                 dec_dim_list=dec_dim_list, para_1=para_1, para_2=para_2, para_3=para_3, model_path=model_path,
                 restore_path=restore_path)

    nmi_mean, acc_mean, fscore_mean, RI_mean, pred_, true_ = train_AE(x, gt, AE, num_class, sample_num, num_views)

    print("para_1:%f, para_2:%f, para_3:%f" % (para_1, para_2, para_3))

    


            







