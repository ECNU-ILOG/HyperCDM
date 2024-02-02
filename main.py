import json
from pprint import pprint

import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
import scipy.sparse as sp
import numbers


from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.cluster import KMeans
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from DOA import DOA
from collections import OrderedDict
from joblib import Parallel, delayed


class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim, n_clusters):
        super(AutoEncoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = self.input_dim
        self.hidden_dims = hidden_dims
        self.hidden_dims.append(latent_dim)
        self.dims_list = (hidden_dims + hidden_dims[:-1][::-1])  # mirrored structure
        self.n_layers = len(self.dims_list)
        self.latent_dim = latent_dim
        self.n_clusters = n_clusters

        # Validation check
        assert self.n_layers % 2 > 0
        assert self.dims_list[self.n_layers // 2] == self.latent_dim

        # Encoder Network
        layers = OrderedDict()
        for idx, hidden_dim in enumerate(self.hidden_dims):
            if idx == 0:
                layers.update(
                    {
                        'linear0': nn.Linear(self.input_dim, hidden_dim, dtype=torch.float64),
                        'activation0': nn.ReLU()
                    }
                )
            else:
                layers.update(
                    {
                        'linear{}'.format(idx): nn.Linear(
                            self.hidden_dims[idx-1], hidden_dim, dtype=torch.float64),
                        'activation{}'.format(idx): nn.ReLU(),
                        'bn{}'.format(idx): nn.BatchNorm1d(
                            self.hidden_dims[idx], dtype=torch.float64)
                    }
                )
        self.encoder = nn.Sequential(layers)

        # Decoder Network
        layers = OrderedDict()
        tmp_hidden_dims = self.hidden_dims[::-1]
        for idx, hidden_dim in enumerate(tmp_hidden_dims):
            if idx == len(tmp_hidden_dims) - 1:
                layers.update(
                    {
                        'linear{}'.format(idx): nn.Linear(
                            hidden_dim, self.output_dim, dtype=torch.float64),
                    }
                )
            else:
                layers.update(
                    {
                        'linear{}'.format(idx): nn.Linear(
                            hidden_dim, tmp_hidden_dims[idx+1], dtype=torch.float64),
                        'activation{}'.format(idx): nn.ReLU(),
                        'bn{}'.format(idx): nn.BatchNorm1d(
                            tmp_hidden_dims[idx+1], dtype=torch.float64)
                    }
                )
        self.decoder = nn.Sequential(layers)

    def __repr__(self):
        repr_str = '[Structure]: {}-'.format(self.input_dim)
        for idx, dim in enumerate(self.dims_list):
            repr_str += '{}-'.format(dim)
        repr_str += str(self.output_dim) + '\n'
        repr_str += '[n_layers]: {}'.format(self.n_layers) + '\n'
        repr_str += '[n_clusters]: {}'.format(self.n_clusters) + '\n'
        repr_str += '[input_dims]: {}'.format(self.input_dim)
        return repr_str

    def __str__(self):
        return self.__repr__()

    def forward(self, X, latent=False):
        output = self.encoder(X)
        if latent:
            return output
        return self.decoder(output)


def _parallel_compute_distance(X, cluster):
    n_samples = X.shape[0]
    dis_mat = np.zeros((n_samples, 1))
    for i in range(n_samples):
        dis_mat[i] += np.sqrt(np.sum((X[i] - cluster) ** 2, axis=0))
    return dis_mat


class BatchKMeans(object):
    def __init__(self, latent_dim, n_clusters, n_jobs):
        self.n_features = latent_dim
        self.n_clusters = n_clusters
        self.clusters = np.zeros((self.n_clusters, self.n_features))
        self.count = 100 * np.ones((self.n_clusters))  # serve as learning rate
        self.n_jobs = n_jobs

    def _compute_dist(self, X):
        dis_mat = Parallel(n_jobs=self.n_jobs)(
            delayed(_parallel_compute_distance)(X, self.clusters[i])
            for i in range(self.n_clusters))
        dis_mat = np.hstack(dis_mat)

        return dis_mat

    def init_cluster(self, X):
        """ Generate initial clusters using sklearn.Kmeans """
        model = KMeans(n_clusters=self.n_clusters, n_init=20)
        model.fit(X)
        self.clusters = model.cluster_centers_  # copy clusters

    def update_cluster(self, X, cluster_idx):
        """ Update clusters in Kmeans on a batch of data """
        n_samples = X.shape[0]
        for i in range(n_samples):
            self.count[cluster_idx] += 1
            eta = 1.0 / self.count[cluster_idx]
            updated_cluster = ((1 - eta) * self.clusters[cluster_idx] +
                               eta * X[i])
            self.clusters[cluster_idx] = updated_cluster

    def update_assign(self, X):
        """ Assign samples in `X` to clusters """
        dis_mat = self._compute_dist(X)
        return np.argmin(dis_mat, axis=1)

    def assign_group(self, X, belong):
        dis_mat = self._compute_dist(X)
        return np.argsort(dis_mat, axis=1)[:, :belong]


class DeepClusteringNet(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim, n_clusters, beta=1, lamda=1,
                 pretrained=True, lr=0.0001, device="cuda", n_jobs=-1):
        super(DeepClusteringNet, self).__init__()
        self.beta = beta  # coefficient of the clustering term
        self.lamda = lamda  # coefficient of the reconstruction term
        self.device = device
        self.pretrained = pretrained
        self.n_clusters = n_clusters

        # Validation check
        if not self.beta > 0:
            msg = 'beta should be greater than 0 but got value = {}.'
            raise ValueError(msg.format(self.beta))

        if not self.lamda > 0:
            msg = 'lambda should be greater than 0 but got value = {}.'
            raise ValueError(msg.format(self.lamda))

        if len(hidden_dims) == 0:
            raise ValueError('No hidden layer specified.')

        self.kmeans = BatchKMeans(latent_dim, n_clusters, n_jobs)
        self.autoencoder = AutoEncoder(input_dim, hidden_dims, latent_dim, n_clusters).to(self.device)

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(),
                                          lr=lr,
                                          weight_decay=5e-4)

    def _loss(self, X, cluster_id):
        batch_size = X.size()[0]
        rec_X = self.autoencoder(X)
        latent_X = self.autoencoder(X, latent=True)

        # Reconstruction error
        rec_loss = self.lamda * self.criterion(X, rec_X)

        # Regularization term on clustering
        dist_loss = torch.tensor(0.).to(self.device)
        clusters = torch.FloatTensor(self.kmeans.clusters).to(self.device)
        for i in range(batch_size):
            diff_vec = latent_X[i] - clusters[cluster_id[i]]
            sample_dist_loss = torch.matmul(diff_vec.view(1, -1),
                                            diff_vec.view(-1, 1))
            dist_loss += 0.5 * self.beta * torch.squeeze(sample_dist_loss)

        return (rec_loss + dist_loss,
                rec_loss.detach().cpu().numpy(),
                dist_loss.detach().cpu().numpy())

    def pretrain(self, train_loader, epoch=100):
        if not self.pretrained:
            return
        if not isinstance(epoch, numbers.Integral):
            msg = '`epoch` should be an integer but got value = {}'
            raise ValueError(msg.format(epoch))

        rec_loss_list = []

        self.train()
        for e in tqdm(range(epoch), "gain feature"):
            for data in train_loader:
                batch_size = data.size()[0]
                data = data.to(self.device).view(batch_size, -1)
                rec_X = self.autoencoder(data)
                loss = self.criterion(data, rec_X)

                rec_loss_list.append(loss.detach().cpu().numpy())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        self.eval()

        # Initialize clusters in self.kmeans after pre-training
        batch_X = []
        for data in train_loader:
            batch_size = data.size()[0]
            data = data.to(self.device).view(batch_size, -1)
            latent_X = self.autoencoder(data, latent=True)
            batch_X.append(latent_X.detach().cpu().numpy())
        batch_X = np.vstack(batch_X)
        self.kmeans.init_cluster(batch_X)

        return rec_loss_list

    def fit(self, train_loader, epoch=50):
        for e in tqdm(range(epoch), "clustering"):
            self.train()
            for data in train_loader:
                batch_size = data.size()[0]
                data = data.view(batch_size, -1).to(self.device)

                # Get the latent features
                with torch.no_grad():
                    latent_X = self.autoencoder(data, latent=True)
                    latent_X = latent_X.cpu().numpy()

                # [Step-1] Update the assignment results
                cluster_id = self.kmeans.update_assign(latent_X)

                # [Step-2] Update clusters in bath Kmeans
                elem_count = np.bincount(cluster_id,
                                         minlength=self.n_clusters)
                for k in range(self.n_clusters):
                    # avoid empty slicing
                    if elem_count[k] == 0:
                        continue
                    self.kmeans.update_cluster(latent_X[cluster_id == k], k)

                # [Step-3] Update the network parameters
                loss, rec_loss, dist_loss = self._loss(data, cluster_id)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def gain_clusters(self, train_loader, belong):
        clusters = []
        for data in train_loader:
            batch_size = data.size()[0]
            data = data.view(batch_size, -1).to(self.device)

            with torch.no_grad():
                latent_X = self.autoencoder(data, latent=True)
                latent_X = latent_X.cpu().numpy()

            cluster_id = self.kmeans.assign_group(latent_X, belong)
            clusters.append(cluster_id)
        group_ids = np.vstack(clusters)
        return group_ids


class Hypergraph:
    def __init__(self, H: np.ndarray):
        self.H = H
        # avoid zero
        self.Dv = np.count_nonzero(H, axis=1) + 1
        self.De = np.count_nonzero(H, axis=0) + 1

    def to_tensor_nadj(self):
        coo = sp.coo_matrix(self.H @ np.diag(1 / self.De) @ self.H.T @ np.diag(1 / self.Dv))
        indices = torch.from_numpy(np.asarray([coo.row, coo.col]))
        return torch.sparse_coo_tensor(indices, coo.data, coo.shape, dtype=torch.float64).coalesce()


class NoneNegClipper(object):
    def __init__(self):
        super(NoneNegClipper, self).__init__()

    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            a = torch.relu(torch.neg(w))
            w.add_(a)


class ResponseLogs:
    def __init__(self, path, train_size=0.8):
        print("Load dataset config file")
        with open(path+"/config.json") as file:
            config = json.load(file)
        print("Dataset name: {}".format(config["dataset"]))

        self.config = config
        self.q_matrix = read_csv(path+"/"+config["q_file"], header=None).to_numpy()
        self.response_logs = read_csv(path+"/"+config["data"], header=None).to_numpy(dtype=int)
        self.train_set, self.test_set = train_test_split(self.response_logs,
                                                         train_size=int(train_size*self.response_logs.shape[0]))

        print("Load successfully! Logs entry: {}".format(self.response_logs.shape[0]))

    def hyper_construct(self, choice="student"):
        # Only use train set to avoid leakage
        if choice == "student":
            print("Construct student hypergraph")
            """
            strategy: deep clustering ==> hypergraph
            """
            X = torch.tensor(self.get_r_matrix(choice="train"), dtype=torch.float64)
            n_clusters = int(int(self.config["student_num"]) * 0.02)
            student_response_loader = torch.utils.data.DataLoader(dataset=X, batch_size=256, shuffle=False)
            clf = DeepClusteringNet(input_dim=int(self.config["exercise_num"]),
                                    hidden_dims=[512, 256, 128],
                                    latent_dim=64,
                                    n_clusters=n_clusters)
            clf.pretrain(student_response_loader)
            clf.fit(student_response_loader)
            groups = clf.gain_clusters(student_response_loader, n_clusters // 2)
            H = np.zeros((int(self.config["student_num"]), n_clusters))
            for i in range(H.shape[0]):
                H[i, groups[i]] = 1
            H = H[:, np.count_nonzero(H, axis=0) >= 2]  # remove empty edge
        elif choice == "exercise":
            print("Construct exercise hypergraph")
            # strategy one
            H = self.q_matrix.copy()
            H = H[:, np.count_nonzero(H, axis=0) >= 2]  # remove empty edge
        elif choice == "knowledge":
            print("Construct knowledge concept hypergraph")
            # strategy one
            H = self.q_matrix.T.copy()
            H = H[:, np.count_nonzero(H, axis=0) >= 2]  # remove empty edge
        else:
            raise ValueError("Only \"student\", \"exercise\" and \"knowledge\" are capable for parameter choice")
        return Hypergraph(H)

    def transform(self, choice="train", batch_size=32):
        if choice == "train":
            dataset = TensorDataset(
                torch.tensor(self.train_set[:, 0], dtype=torch.int64),
                torch.tensor(self.train_set[:, 1], dtype=torch.int64),
                torch.tensor(self.q_matrix[self.train_set[:, 1], :]),
                torch.tensor(self.train_set[:, 2], dtype=torch.float64)
            )
        elif choice == "test":
            dataset = TensorDataset(
                torch.tensor(self.test_set[:, 0], dtype=torch.int64),
                torch.tensor(self.test_set[:, 1], dtype=torch.int64),
                torch.tensor(self.q_matrix[self.test_set[:, 1], :]),
                torch.tensor(self.test_set[:, 2], dtype=torch.float64)
            )
        else:
            raise ValueError("Only \"train\" and \"test\" are capable for parameter choice")
        return DataLoader(dataset, batch_size, shuffle=True)

    def get_r_matrix(self, choice="train"):
        r_matrix = -1 * np.ones(shape=(int(self.config["student_num"]), int(self.config["exercise_num"])))
        if choice == "train":
            for line in self.train_set:
                student_id = line[0]
                exercise_id = line[1]
                score = line[2]
                r_matrix[student_id, exercise_id] = int(score)
        elif choice == "test":
            for line in self.test_set:
                student_id = line[0]
                exercise_id = line[1]
                score = line[2]
                r_matrix[student_id, exercise_id] = int(score)
        else:
            raise ValueError("Only \"train\" and \"test\" are capable for parameter choice")
        return r_matrix


class HSCD_Net(nn.Module):
    def __init__(self, student_num, exercise_num, knowledge_num, feature_dim, emb_dim,
                 student_adj, exercise_adj, knowledge_adj, device, layers=3, leaky=0.8):
        super(HSCD_Net, self).__init__()

        self.student_num = student_num
        self.exercise_num = exercise_num
        self.knowledge_num = knowledge_num
        self.feature_dim = feature_dim
        self.emb_dim = emb_dim

        self.device = device
        self.layers = layers
        self.leaky = leaky

        self.student_emb = nn.Embedding(student_num, emb_dim, dtype=torch.float64)
        self.exercise_emb = nn.Embedding(exercise_num, emb_dim, dtype=torch.float64)
        self.knowledge_emb = nn.Embedding(knowledge_num, emb_dim, dtype=torch.float64)

        self.student_emb2feature = nn.Linear(emb_dim, feature_dim, dtype=torch.float64)
        self.exercise_emb2feature = nn.Linear(emb_dim, feature_dim, dtype=torch.float64)
        self.knowledge_emb2feature = nn.Linear(emb_dim, feature_dim, dtype=torch.float64)
        self.exercise_emb2discrimination = nn.Linear(emb_dim, 1, dtype=torch.float64)

        self.clipper = NoneNegClipper()

        self.state2response = nn.Sequential(
            nn.Linear(knowledge_num, 512, dtype=torch.float64),
            nn.Tanh(),
            nn.Linear(512, 256, dtype=torch.float64),
            nn.Tanh(),
            nn.Linear(256, 128, dtype=torch.float64),
            nn.Tanh(),
            nn.Linear(128, 1, dtype=torch.float64),
            nn.Sigmoid()
        )

        self.student_adj = student_adj
        self.exercise_adj = exercise_adj
        self.knowledge_adj = knowledge_adj

        # initialize
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def convolution(self, embedding, adj):
        all_emb = embedding.weight.to(self.device)
        final = [all_emb]
        for i in range(self.layers):
            # implement momentum hypergraph convolution
            all_emb = torch.sparse.mm(adj, all_emb) + 0.8 * all_emb
            final.append(all_emb)
        final_emb = torch.mean(torch.stack(final, dim=1), dim=1, dtype=torch.float64)
        return final_emb

    def forward(self, student_id, exercise_id, knowledge):
        convolved_student_emb = self.convolution(self.student_emb, self.student_adj)
        convolved_exercise_emb = self.convolution(self.exercise_emb, self.exercise_adj)
        convolved_knowledge_emb = self.convolution(self.knowledge_emb, self.knowledge_adj)

        batch_student = f.embedding(student_id, convolved_student_emb)
        batch_exercise = f.embedding(exercise_id, convolved_exercise_emb)

        student_feature = f.leaky_relu(self.student_emb2feature(batch_student), negative_slope=self.leaky)
        exercise_feature = f.leaky_relu(self.exercise_emb2feature(batch_exercise), negative_slope=self.leaky)
        knowledge_feature = f.leaky_relu(self.knowledge_emb2feature(convolved_knowledge_emb), negative_slope=self.leaky)
        discrimination = torch.sigmoid(self.exercise_emb2discrimination(batch_exercise))

        state = discrimination * (student_feature @ knowledge_feature.T
                                  - exercise_feature @ knowledge_feature.T) * knowledge

        state = self.state2response(state)
        return state.view(-1)

    def apply_clipper(self):
        for layer in self.state2response:
            if isinstance(layer, nn.Linear):
                layer.apply(self.clipper)

    def get_mastery_level(self):
        convolved_student_emb = self.convolution(self.student_emb, self.student_adj)
        convolved_knowledge_emb = self.convolution(self.knowledge_emb, self.knowledge_adj)

        student_feature = f.leaky_relu(self.student_emb2feature(convolved_student_emb), negative_slope=self.leaky)
        knowledge_feature = f.leaky_relu(self.knowledge_emb2feature(convolved_knowledge_emb), negative_slope=self.leaky)

        return torch.sigmoid(student_feature @ knowledge_feature.T).detach().cpu().numpy()

    def get_exercise_level(self):
        convolved_exercise_emb = self.convolution(self.exercise_emb, self.exercise_adj)
        convolved_knowledge_emb = self.convolution(self.knowledge_emb, self.knowledge_adj)

        exercise_feature = f.leaky_relu(self.exercise_emb2feature(convolved_exercise_emb), negative_slope=self.leaky)
        knowledge_feature = f.leaky_relu(self.knowledge_emb2feature(convolved_knowledge_emb), negative_slope=self.leaky)

        return torch.sigmoid(exercise_feature @ knowledge_feature.T).detach().cpu().numpy()

    def get_knowledge_feature(self):
        convolved_knowledge_emb = self.convolution(self.knowledge_emb, self.knowledge_adj)
        knowledge_feature = f.leaky_relu(self.knowledge_emb2feature(convolved_knowledge_emb), negative_slope=self.leaky)
        return knowledge_feature.detach().cpu().numpy()


class HyperCDM:
    def __init__(self, student_num, exercise_num, knowledge_num, feature_dim, emb_dim=64, layers=4,
                 device="cpu"):
        self.net: HSCD_Net

        self.student_num = student_num
        self.exercise_num = exercise_num
        self.knowledge_num = knowledge_num
        self.feature_dim = feature_dim
        self.emb_dim = emb_dim
        self.layers = layers

        self.student_hyper = None
        self.exercise_hyper = None
        self.knowledge_hyper = None

        self.device = device

    def train(self, train_set, valid_set=None, q_matrix=None, r_matrix=None, epoch=4, lr=0.0001):
        # train is transformed
        if self.student_hyper is None or self.exercise_hyper is None or self.knowledge_hyper is None:
            raise RuntimeError("Use hyperbuild() method first")
        self.net = HSCD_Net(self.student_num, self.exercise_num, self.knowledge_num,
                            self.feature_dim, self.emb_dim,
                            student_adj=self.student_hyper.to_tensor_nadj().to(self.device),
                            exercise_adj=self.exercise_hyper.to_tensor_nadj().to(self.device),
                            knowledge_adj=self.knowledge_hyper.to_tensor_nadj().to(self.device),
                            layers=self.layers,
                            device=self.device)
        self.net.to(self.device)
        bce_loss_function = nn.BCELoss()
        optimizer = optim.Adam(self.net.parameters(), lr=lr, weight_decay=0.0005)
        total_params = sum(p.numel() for p in self.net.parameters())
        print("Total number of parameters of HGCDM: {}".format(total_params))

        for epoch_i in range(epoch):
            self.net.train()

            epoch_losses = []
            for batch_data in tqdm(train_set, "Epoch {}".format(epoch_i + 1)):
                student_id, exercise_id, knowledge, y = batch_data
                student_id: torch.Tensor = student_id.to(self.device)
                exercise_id: torch.Tensor = exercise_id.to(self.device)
                knowledge = knowledge.to(self.device)
                y: torch.Tensor = y.to(self.device)
                pred_y = self.net.forward(student_id, exercise_id, knowledge)
                bce_loss = bce_loss_function(pred_y, y)
                optimizer.zero_grad()
                bce_loss.backward()
                optimizer.step()
                self.net.apply_clipper()
                epoch_losses.append(bce_loss.mean().item())
            print("[Epoch %d] average loss: %.6f" % (epoch_i + 1, float(np.mean(epoch_losses))))

            if valid_set is not None:
                pprint(self.eval(valid_set, q_matrix, r_matrix))

    def eval(self, test_set, q_matrix=None, r_matrix=None):
        self.net = self.net.to(self.device)
        self.net.eval()
        y_true, y_pred = [], []
        mastery = self.net.get_mastery_level()
        for batch_data in tqdm(test_set, "Evaluating"):
            student_id, exercise_id, knowledge, y = batch_data
            student_id: torch.Tensor = student_id.to(self.device)
            exercise_id: torch.Tensor = exercise_id.to(self.device)
            knowledge = knowledge.to(self.device)
            y: torch.Tensor = y.to(self.device)
            pred_y = self.net.forward(student_id, exercise_id, knowledge)
            y_pred.extend(pred_y.detach().cpu().tolist())
            y_true.extend(y.tolist())
        auc = roc_auc_score(y_true, y_pred)
        acc = accuracy_score(y_true, np.array(y_pred) >= 0.5)
        f1 = f1_score(y_true, np.array(y_pred) >= 0.5)
        if q_matrix is None or r_matrix is None:
            print("Evaluation AUC: %.6f. ACC: %.6f. F1: %.6f" % (auc, acc, f1))
            return {"auc": auc, "acc": acc, "f1": f1}
        else:
            doa = DOA(mastery, q_matrix, r_matrix)
            print("Evaluation AUC: %.6f. ACC: %.6f. F1: %.6f. DOA: %.6f" % (auc, acc, f1, doa))
            return {"auc": auc, "acc": acc, "f1": f1, "doa": doa}

    def hyper_build(self, response_logs):
        self.student_hyper = response_logs.hyper_construct("student")
        self.exercise_hyper = response_logs.hyper_construct("exercise")
        self.knowledge_hyper = response_logs.hyper_construct("knowledge")


response_logs = ResponseLogs("data/Math1")
config = response_logs.config
cdm = HyperCDM(int(config["student_num"]), int(config["exercise_num"]), int(config["knowledge_num"]), 512, 16, 5, device="cuda")
cdm.hyper_build(response_logs)
cdm.train(response_logs.transform(choice="train", batch_size=64),
          response_logs.transform(choice="test", batch_size=64),
          q_matrix=response_logs.q_matrix,
          r_matrix=response_logs.get_r_matrix(choice="test"),
          epoch=10)

