import os
import time
import torch
import datetime
import numpy as np
from common.utils import multi_acc, load_bert_dataset, load_net, bertTokenizer


class BTrainer(object):
    def __init__(self, config):
        self.config = config
        config.encoder = 'bert'
        self.train_itr, self.dev_itr, self.test_itr = load_bert_dataset(config)
        self.encoder, self.net, self.optim = load_net(config)
        self.early_stop = config.TRAIN.early_stop
        self.best_dev_acc = 0
        self.unimproved_iters = 0
        self.iters_not_improved = 0

    def train_epoch(self):
        loss_fn = torch.nn.CrossEntropyLoss()
        metric_fn = multi_acc
        total_loss = []
        total_acc = []
        for step, batch in enumerate(self.train_itr):
            start_time = time.time()
            self.optim.zero_grad()
            text, mask, label = bertTokenizer(batch, self.config)
            feature = self.encoder(text, mask=mask)
            logits, views, atts, atts_logits = self.net(feature, mask=mask)
            # logits = self.encoder(text, mask=mask)
            loss = loss_fn(logits, label)
            metric = metric_fn(label, logits)
            loss.backward()
            self.optim.step()

            total_loss.append(loss.item())
            total_acc.append(metric.item())

            # monitoring results on every steps
            end_time = time.time()
            span_time = (end_time - start_time) * (
                    int(len(self.train_itr.dataset) / self.config.TRAIN.batch_size) - step)
            h = span_time // (60 * 60)
            m = (span_time % (60 * 60)) // 60
            s = (span_time % (60 * 60)) % 60 // 1
            print(
                "\rIteration: {:>4}/{} ({:>4.1f}%)  Loss: {:.5f} -ETA {:>2}h-{:>2}m-{:>2}s".format(
                    step, int(len(self.train_itr.dataset) / self.config.TRAIN.batch_size),
                    100 * (step) / int(len(self.train_itr.dataset) / self.config.TRAIN.batch_size),
                    loss,
                    int(h), int(m), int(s)),
                end="".center(20, ' '))

        return np.array(total_loss).mean(), np.array(total_acc).mean()

    def train(self):
        logfile = open(
            self.config.log_path +
            '/log_run_' + self.config.dataset + '_' + self.config.version + '.txt',
            'a+'
        )
        logfile.write(
            'nowTime: ' +
            datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') +
            '\n'
        )
        logfile.close()
        self.encoder.eval()
        for epoch in range(0, self.config.TRAIN.max_epoch):
            self.encoder.train()
            self.net.train()
            train_loss, train_acc = self.train_epoch()

            logs = ("    Epoch:{:>2}    ".format(epoch)).center(50, "-") + ' '*10 +'\n' + \
                   self.get_logging(train_loss, train_acc, "training")
            print("\r" + logs)

            # logging training logs
            self.logging(self.config.log_path + '/log_run_' + self.config.dataset + '_' + self.config.version + '.txt',
                         logs)

            # saving state
            state = {
                'encoder': self.encoder.state_dict(),
                'state_dict': self.net.state_dict(),
            }

            self.net.eval()
            self.encoder.eval()
            eval_loss, eval_acc = self.eval(self.test_itr, state=state)
            eval_logs = self.get_logging(eval_loss, eval_acc, "evaluating")
            print("\r" + eval_logs)

            # logging evaluating logs
            self.logging(self.config.log_path + '/log_run_' + self.config.dataset + '_' + self.config.version + '.txt',
                         eval_logs)

            # early stopping
            if eval_acc > self.best_dev_acc:
                self.unimproved_iters = 0
                self.best_dev_acc = eval_acc
                # saving models
                torch.save(
                    state,
                    self.config.ckpts_path + '/ckpt_{}_{}.pkl'.format(self.config.dataset, self.config.version)
                )

            else:
                self.unimproved_iters += 1
                if self.unimproved_iters >= self.config.TRAIN.patience and self.early_stop == True:
                    print(
                        self.config.log_path + '/log_run_' + self.config.dataset + '_' + self.config.version + '.txt',
                        "Early Stopping. Epoch: {}, Best Dev Acc: {}".format(epoch, self.best_dev_acc),
                    )
                    break

    def eval(self, eval_itr, state=None):
        # loading models
        if state is not None:
            self.encoder.load_state_dict(state['encoder'])
            self.net.load_state_dict(state['state_dict'])
        else:
            try:
                state = torch.load(
                    self.config.ckpts_path + '/ckpt_{}_{}.pkl'.format(self.config.dataset, self.config.version)
                )
                self.encoder.load_state_dict(state['encoder'])
                self.net.load_state_dict(state['state_dict'])
            except:
                print("can't find the path to load state_dict from pretrained model!")
                exit()

        loss_fn = torch.nn.CrossEntropyLoss()
        metric_fn = multi_acc
        total_loss = []
        total_acc = []
        for step, batch in enumerate(eval_itr):
            start_time = time.time()
            text, mask, label = bertTokenizer(batch, self.config)
            feature = self.encoder(text, mask=mask)
            # logits
            logits, views, atts, atts_logits = self.net(feature, mask=mask)
            # logits = self.encoder(text, mask=mask)
            loss = loss_fn(logits, label)
            metric = metric_fn(label, logits)

            total_loss.append(loss.item())
            total_acc.append(metric.item())

            # monitoring results on every steps
            end_time = time.time()
            span_time = (end_time - start_time) * (
                    int(len(eval_itr.dataset) / self.config.TRAIN.batch_size) - step)
            h = span_time // (60 * 60)
            m = (span_time % (60 * 60)) // 60
            s = (span_time % (60 * 60)) % 60 // 1
            print(
                "\rIteration: {:>4}/{} ({:>4.1f}%)   -ETA {:>2}h-{:>2}m-{:>2}s".format(
                    step, int(len(eval_itr.dataset) / self.config.TRAIN.batch_size),
                    100 * (step) / int(len(eval_itr.dataset) / self.config.TRAIN.batch_size),
                    int(h), int(m), int(s)),
                end="".center(20, ' '))

        return np.array(total_loss).mean(), np.array(total_acc).mean()

    def run(self, run_mode):
        if run_mode == 'train':
            self.empty_log(self.config.version)
            self.train()
        else:
            exit(-1)

    def empty_log(self, version):
        if (os.path.exists(self.config.log_path + '/log_run_' + self.config.dataset + '_' + version + '.txt')):
            os.remove(self.config.log_path + '/log_run_' + self.config.dataset + '_' + version + '.txt')
        print('Initializing log file ........')
        print('Finished!')
        print('')

    def get_logging(self, loss, acc, eval='training'):
        logs = \
            '==={} phrase...'.format(eval) + ' '*50 + '\n' + \
            'loss :{:<10.3f}\t acc :{:<10.3f}'.format(loss, acc) + '\n'
        return logs

    def logging(self, log_file, logs):
        logfile = open(
            log_file, 'a+'
        )
        logfile.write(logs)
        logfile.close()