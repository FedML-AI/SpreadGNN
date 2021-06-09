import copy
import logging

import numpy as np
import torch
import wandb
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from scipy.linalg import fractional_matrix_power
from torch import nn


class DecentralizedWorker(object):
    def __init__(self, worker_index, topology_manager, train_data_local_dict, test_data_local_dict,
                 train_data_local_num_dict, train_data_num, device, model, args):
        self.round_index = 0

        # topology management
        self.worker_index = worker_index
        self.in_neighbor_idx_list = topology_manager.get_in_neighbor_idx_list(worker_index)
        logging.info("in_neighbor_idx_list (index = %d) = %s" % (self.worker_index, str(self.in_neighbor_idx_list)))

        self.model_dict = dict()
        self.sample_num_dict = dict()
        self.flag_neighbor_result_received_dict = dict()
        for neighbor_idx in self.in_neighbor_idx_list:
            self.flag_neighbor_result_received_dict[neighbor_idx] = False

        # dataset
        self.train_data_local_dict = train_data_local_dict
        self.train_data_local_num_dict = train_data_local_num_dict
        self.test_data_local_dict = test_data_local_dict
        self.all_train_data_num = train_data_num
        self.train_local = self.train_data_local_dict[worker_index]
        self.local_sample_number = self.train_data_local_num_dict[worker_index]

        # model and optimization
        self.device = device
        self.args = args
        self.model = model
        # logging.info(self.model)
        self.model.to(self.device)
        self.criterion = self.args.loss.to(self.device)
        if self.args.client_optimizer == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.wd)
        else:
            self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                              lr=self.args.lr,
                                              weight_decay=self.args.wd, amsgrad=True)
        self.lambda_relationship = args.task_reg

        # initialize the task specific weights
        self.neighbor_task_specific_weight_dict = dict()
        #For GNNs, this is the output layer of the readout
        task_specific_layer = self.model.readout.output
        torch.nn.init.xavier_uniform_(task_specific_layer.weight)

        # correlation matrix
        num_of_neighbors = len(self.in_neighbor_idx_list)
        self.corr_matrix_omega = None

        # test result
        self.train_acc_dict = []
        self.train_loss_dict = []
        self.test_acc_dict = []
        self.test_loss_dict = []
        self.flag_neighbor_test_result_received_dict = dict()
        for neighbor_idx in self.in_neighbor_idx_list:
            self.flag_neighbor_test_result_received_dict[neighbor_idx] = False

    def add_neighbor_local_result(self, index, model_params, neighbor_cli_task_idxs, sample_num):
        # logging.info("add_model. index = %d" % index)
        self.model_dict[index] = model_params
        self.sample_num_dict[index] = sample_num
        self.flag_neighbor_result_received_dict[index] = True
        if self.args.is_mtl == 1:
            # Note: Loss doesn't backprop through copy-based reshapes https://github.com/pytorch/xla/issues/870
            #CHECK
            task_specific_weight = model_params['readout.output.weight'][neighbor_cli_task_idxs, :] # Of shape 27 x 64
            
            # logging.info("task_specific_weight = " + str(task_specific_weight))
            self.neighbor_task_specific_weight_dict[index] = task_specific_weight.to(self.device)

    def check_whether_all_receive(self):
        for neighbor_idx in self.in_neighbor_idx_list:
            if not self.flag_neighbor_result_received_dict[neighbor_idx]:
                return False
        for neighbor_idx in self.in_neighbor_idx_list:
            self.flag_neighbor_result_received_dict[neighbor_idx] = False
        return True

    def calculate_relationship_regularizer_with_trace(self, client_task_idxs):
        # for the first round, since there is no exchange information among workers, we do not need to add relationship
        if self.round_index == 0:
            return 0.0
        tensor_list = []
        # update local specific weights
        task_specific_weight = self.model.readout.output.weight[client_task_idxs, :]
        self.neighbor_task_specific_weight_dict[self.worker_index] = task_specific_weight

        # logging.info("neighbor len = %d" % len(self.neighbor_task_specific_weight_dict))
        
        tensor_list.append(task_specific_weight.detach())
        for neighbor_idx in self.in_neighbor_idx_list:
            tensor_list.append(self.neighbor_task_specific_weight_dict[neighbor_idx])
            # logging.info("worker_index = %d, require_grad = %d" % (self.worker_index,
            #                                  self.neighbor_task_specific_weight_dict[neighbor_idx].requires_grad))
        
        weight_matrix = torch.cat(tensor_list, dim=0) # 27 x 64 dimensional
        trans_w = torch.transpose(weight_matrix, 0, 1) # 64 x 27 dimensional
        if self.corr_matrix_omega is None:
            self.corr_matrix_omega = torch.ones(weight_matrix.shape[0], weight_matrix.shape[0], device=self.device, requires_grad=False)

        # (H, N_nb) * (N_nb, N_nb) * (N_nb, H)
        # logging.info("self.corr_matrix_omega = " + str(self.corr_matrix_omega))
        relationship_trace = torch.trace(torch.matmul(trans_w, torch.matmul(self.corr_matrix_omega, weight_matrix)))
        return relationship_trace

    def update_correlation_matrix(self, client_task_idxs):
        if self.round_index == 0:
            return
        tensor_list = []
        task_specific_weight = self.model.readout.output.weight[client_task_idxs, :]
        self.neighbor_task_specific_weight_dict[self.worker_index] = task_specific_weight

        tensor_list.append(task_specific_weight.detach())
        for neighbor_idx in self.in_neighbor_idx_list:
            tensor_list.append(self.neighbor_task_specific_weight_dict[neighbor_idx].detach())

        
        weight_matrix = torch.cat(tensor_list, dim=0) # 27 x 64 dimensional
        trans_w = torch.transpose(weight_matrix, 0, 1) # 64 x 27
        corr_trans = torch.matmul(weight_matrix, trans_w) # 27 x 27
        # logging.info(corr_trans)
        corr_new_np = fractional_matrix_power(corr_trans.cpu(), 1 / 2)
        self.corr_matrix_omega = torch.from_numpy(corr_new_np / np.trace(corr_new_np)).float().to(self.device)
        self.corr_matrix_omega.requires_grad = False

    def aggregate(self):
        model_list = []
        training_num = 0

        for neighbor_idx in self.in_neighbor_idx_list:
            model_list.append((self.sample_num_dict[neighbor_idx], self.model_dict[neighbor_idx]))
            training_num += self.sample_num_dict[neighbor_idx]

        # logging.info("len of self.model_dict[idx] = " + str(len(self.model_dict)))

        # logging.info("################aggregate: %d" % len(model_list))
        (num0, averaged_params) = model_list[0]
        for k in averaged_params.keys():
            # only update the shared layers
            #CHECK HERE
            if self.args.is_mtl == 1 and k == "task_specific_layer.weight" or k == "task_specific_layer.bias":
                continue
            for i in range(0, len(model_list)):
                local_sample_number, local_model_params = model_list[i]
                w = local_sample_number / training_num
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] += local_model_params[k] * w

        # update the global model which is cached at the server side
        self.model.load_state_dict(averaged_params)

    def train(self, round_index):
        self.round_index = round_index
        self.model.to(self.device)
        # change to train mode
        self.model.train()
        mask_check = 0 if isinstance(self.criterion,(type(torch.nn.MSELoss()), type(torch.nn.L1Loss()))) else 1

        for epoch in range(self.args.epochs):
            #Molecule is single batch
            for mol_idxs, (adj_matrix, feature_matrix, label, mask, cli_mask) in enumerate(self.train_local):
                self.cli_mask_idxs = cli_mask.flatten().nonzero(as_tuple=False).flatten()

                # iterative step 1: update the theta, W_i
                # Pass on molecules that have no labels
                mask = mask.to(device=self.device, dtype=torch.float32, non_blocking=True) # Always a tensor of ones for regression 
                cli_mask = cli_mask.to(device=self.device, dtype=torch.float32, non_blocking=True) if cli_mask is not None else None
                mask = mask * cli_mask if cli_mask is not None else mask
                if torch.all(mask == 0).item():
                    continue

                self.optimizer.zero_grad()
                if self.args.model =="graphsage":
                     adj_matrix= [level.to(device=self.device, dtype=torch.long, non_blocking=True) for level in adj_matrix]
                else:
                    adj_matrix = adj_matrix.to(device= self.device, dtype=torch.float32, non_blocking=True)
                feature_matrix = feature_matrix.to(device= self.device, dtype=torch.float32, non_blocking=True)
                label = label.to(device=self.device, dtype=torch.float32, non_blocking=True)
                
                log_probs = self.model(adj_matrix, feature_matrix)
                predict_loss = self.criterion(log_probs, label) * mask
                predict_loss = predict_loss.sum() / mask.sum()

                if self.args.is_mtl == 1:
                    relationship_trace = self.calculate_relationship_regularizer_with_trace(self.cli_mask_idxs)
                    total_loss = predict_loss + self.lambda_relationship * relationship_trace
                    total_loss = total_loss.sum() / mask.sum()
                    # batch_loss.append(total_loss.item())
                    total_loss.backward()
                else:
                    predict_loss = predict_loss.sum() / mask.sum()
                    # batch_loss.append(predict_loss.item().numpy())
                    predict_loss.backward()

                self.optimizer.step()

                # iterative step 2: update relationship matrix omega
                if self.args.is_mtl == 1:
                    self.update_correlation_matrix(self.cli_mask_idxs)

            # epoch_loss.append(sum(batch_loss) / len(batch_loss))
            # logging.info('(client {}. Local Training Epoch: {} \tLoss: {:.6f}'.format(self.worker_index,
            #                                                                           epoch, sum(epoch_loss) / len(
            #                                                                                   epoch_loss)))
            

        weights = self.model.cpu().state_dict()
        self.lambda_relationship *= self.args.task_reg_decay
        return weights, self.local_sample_number

    def save_omega(self):
        omega_cpu = self.corr_matrix_omega.detach().cpu().numpy()
        filename = "omega_{}".format(self.worker_index)
        np.save(filename, omega_cpu)

    def test_on_local_data(self, round_idx):
        if round_idx % self.args.frequency_of_the_test == 0 or round_idx == self.args.comm_round - 1:
            logging.info("################local_test_on_all_clients : {}".format(round_idx))
            # train data
            # train_score, train_loss = self._infer(self.train_data_local_dict[self.worker_index])

            # test data
            test_score, test_loss = self._infer(self.test_data_local_dict[self.worker_index])
            logging.info("testscore = %d, test_loss = %d" % (
                test_score, test_loss))

            logging.info("worker_index = %d, test_score = %f, test_loss = %f" % (
                    self.worker_index, test_score, test_loss))
            return test_score, test_loss
        else:
            return None, None

    def _infer(self, test_data):
 
        if isinstance(self.args.loss , type(torch.nn.BCEWithLogitsLoss())):
            return self._infer_clf(test_data)
        else:
            return self._infer_reg(test_data)

        #return score,  test_loss

    def _check_whether_all_test_result_receive(self):
        for neighbor_idx in self.in_neighbor_idx_list:
            if not self.flag_neighbor_test_result_received_dict[neighbor_idx]:
                return False
        for neighbor_idx in self.in_neighbor_idx_list:
            self.flag_neighbor_test_result_received_dict[neighbor_idx] = False
        return True

    def record_average_test_result(self, index, round_idx, test_acc, test_loss):
        self.test_acc_dict.append(test_acc)
        self.test_loss_dict.append(test_loss)
        self.flag_neighbor_test_result_received_dict[index] = True
        if self._check_whether_all_test_result_receive():
            logging.info("ROUND INDEX = %d" % round_idx)

            test_acc = sum(self.test_acc_dict) / len(self.test_acc_dict)
            test_loss = sum(self.test_loss_dict) / len(self.test_loss_dict)
            wandb.log({"Test/Acc": test_acc, "round": round_idx})
            wandb.log({"Test/Loss": test_loss, "round": round_idx})
            stats = {'test_acc': test_acc, 'test_loss': test_loss}
            logging.info(stats)

            self.test_acc_dict.clear()
            self.test_loss_dict.clear()

    def _infer_clf(self, test_data):
        logging.info("----------classification test--------")
        self.model.eval()
        self.model.to(self.device)
        test_loss = 0.
        with torch.no_grad():
            
            y_pred = []
            y_true = []
            masks = []
            for mol_idx, (adj_matrix, feature_matrix, label, mask , _) in enumerate(test_data):
                if self.args.model == "graphsage":
                    adj_matrix = [level.to(device=self.device, dtype=torch.long, non_blocking=True) for level in adj_matrix]
                else:
                    adj_matrix = adj_matrix.to(device=self.device, dtype=torch.float32, non_blocking=True)
                feature_matrix = feature_matrix.to(device=self.device, dtype=torch.float32, non_blocking=True)
                mask = mask.to(device= self.device, dtype=torch.float32, non_blocking=True)
                label = label.to(device= self.device, dtype=torch.float32, non_blocking=True)
                logits = self.model(adj_matrix, feature_matrix)
                loss = self.criterion(logits, label) * mask
                test_loss += loss.sum() / mask.sum()

                y_pred.append(logits.cpu().numpy())
                y_true.append(label.cpu().numpy())
                masks.append(mask.cpu().numpy())

        y_pred = np.array(y_pred)
        y_true = np.array(y_true)
        masks = np.array(masks)

        results = []
        for label in range(masks.shape[1]):
            valid_idxs = np.nonzero(masks[:, label])
            truth = y_true[valid_idxs, label].flatten()
            pred = y_pred[valid_idxs, label].flatten()

            if np.all(truth == 0.0) or np.all(truth == 1.0):
                results.append(float('nan'))
            else:
                if self.args.metric == 'prc-auc':
                    precision, recall, _ = precision_recall_curve(truth, pred)
                    score = auc(recall, precision)
                else:
                    score = roc_auc_score(truth, pred)

                results.append(score)

        score = np.nanmean(results)

        return score, test_loss

    def _infer_reg(self, test_data):
        logging.info("---------- regression test--------")
        self.model.eval()
        self.model.to(self.device)

        
        test_loss = 0.
        with torch.no_grad():
            
            y_pred = []
            y_true = []
            for mol_idx, (forest, feature_matrix, label, _ , _) in enumerate(test_data):
                if self.args.model == "graphsage":
                    forest = [level.to(device= self.device, dtype=torch.long, non_blocking=True) for level in forest]
                else:
                    forest = forest.to(device= self.device, dtype=torch.float32, non_blocking=True)
                label = label.to(device= self.device, dtype=torch.float32, non_blocking=True)
                feature_matrix = feature_matrix.to(device=self.device, dtype=torch.float32, non_blocking=True)
                logits = self.model(forest, feature_matrix)
                test_loss += self.criterion(logits, label).mean().item()
                y_pred.append(logits.cpu().numpy())
                y_true.append(label.cpu().numpy())

            # logging.info(y_true)
            # logging.info(y_pred)
            if self.args.metric == 'rmse':
                score = mean_squared_error(np.array(y_true), np.array(y_pred), squared=False)
            elif self.args.metric == 'r2':
                score = r2_score(np.array(y_true), np.array(y_pred))
            else:
                score = mean_absolute_error(np.array(y_true), np.array(y_pred))

        return score, test_loss


