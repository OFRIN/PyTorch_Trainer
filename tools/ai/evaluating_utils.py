import sys
import torch
import numpy as np

from tools.ai import torch_utils
from tools.general import io_utils, time_utils

class Evaluator:
    def __init__(self, model, loader, class_names, task, amp):
        self.amp = amp
        self.task = task

        self.model = model
        self.loader = loader

        if self.task == 'multi-labels':
            self.evaluator = Evaluator_For_Multi_Label_Classification(class_names)
        else:
            self.evaluator = Evaluator_For_Mean_Accuracy(class_names)

        self.eval_timer = time_utils.Timer()
        self.num_iterations = len(self.loader)

    def step(self, detail=False):
        self.model.eval()
        self.eval_timer.tik()

        with torch.no_grad():
            ni_digits = io_utils.get_digits_in_number(self.num_iterations)
            
            for i, (images, labels) in enumerate(self.loader):
                i += 1
                progress_format = '\r# Evaluation [%0{}d/%0{}d] = %02.2f%%'.format(ni_digits, ni_digits)
            
                sys.stdout.write(progress_format%(i, self.num_iterations, i / self.num_iterations * 100))
                sys.stdout.flush()
                
                if torch.cuda.is_available():
                    images = images.cuda()
                    labels = labels.cuda()
                
                with torch.cuda.amp.autocast(enabled=self.amp):
                    logits = self.model(images)
                
                if self.task == 'multi-labels':
                    preds = torch.sigmoid(logits)
                else:
                    preds = torch.max(logits, 1)[1]

                preds = torch_utils.get_numpy_from_tensor(preds)
                labels = torch_utils.get_numpy_from_tensor(labels)
                
                for i in range(images.size()[0]):
                    self.evaluator.add(preds[i], labels[i])
        print('\r', end='')

        self.model.train()
        return self.evaluator.get(detail=detail), self.eval_timer.tok(clear=True)

class Evaluator_For_Mean_Accuracy:
    def __init__(self, class_names):
        self.class_names = class_names
        self.num_classes = len(self.class_names)

        self.clear()

    def add(self, pred, label):
        self.accuracy_list[label].append(pred == label)

    def get(self, detail=False, clear=True):
        accuracy_dict = {name:np.mean(self.accuracy_list[label]) * 100 for name, label in zip(self.class_names, range(self.num_classes))}
        mean_accuracy = np.mean(list(accuracy_dict.values()))

        if clear:
            self.clear()
        
        if detail:
            return mean_accuracy, accuracy_dict
        else:
            return mean_accuracy

    def clear(self):
        self.accuracy_list = [[] for _ in range(self.num_classes)]

class Evaluator_For_Multi_Label_Classification:
    def __init__(self, class_names, th_interval=0.1):
        self.thresholds = list(np.arange(0.01, 1.00, th_interval))

        self.class_names = class_names
        self.num_classes = len(self.class_names)
        
        self.clear()

    def add(self, pred, gt):
        for th in self.thresholds:
            binary_pred = (pred >= th).astype(np.float32)

            self.meter_dic[th]['P'] += binary_pred
            self.meter_dic[th]['T'] += gt
            self.meter_dic[th]['TP'] += (gt * (binary_pred == gt)).astype(np.float32)

    def get(self, detail=False, clear=True):
        op_list = []
        or_list = []
        o_f1_list = []

        cp_list = []
        cr_list = []
        c_f1_list = []

        for th in self.thresholds:
            data = self.meter_dic[th]

            P = data['P']
            T = data['T']
            TP = data['TP']

            overall_precision = np.sum(TP) / (np.sum(P) + 1e-5) * 100
            overall_recall = np.sum(TP) / (np.sum(T) + 1e-5) * 100
            overall_f1_score = 2 * ((overall_precision * overall_recall) / (overall_precision + overall_recall + 1e-5))

            op_list.append(overall_precision)
            or_list.append(overall_recall)
            o_f1_list.append(overall_f1_score)

            per_class_precision = np.mean(TP / (P + 1e-5)) * 100
            per_class_recall = np.mean(TP / (T + 1e-5)) * 100
            per_class_f1_score = 2 * ((per_class_precision * per_class_recall) / (per_class_precision + per_class_recall + 1e-5))

            cp_list.append(per_class_precision)
            cr_list.append(per_class_recall)
            c_f1_list.append(per_class_f1_score)

        best_index = np.argmax(o_f1_list)
        best_o_th = self.thresholds[best_index]

        best_op = op_list[best_index]
        best_or = or_list[best_index]
        best_of = o_f1_list[best_index]

        best_index = np.argmax(c_f1_list)
        best_c_th = self.thresholds[best_index]
        
        best_cp = cp_list[best_index]
        best_cr = cr_list[best_index]
        best_cf = c_f1_list[best_index]

        if clear:
            self.clear()
        
        if detail:
            return [best_c_th, best_cp, best_cr, best_cf], cp_list, cr_list, c_f1_list
        else:
            return best_cf
    
    def clear(self):
        self.meter_dic = {
            th : {
                'P':np.zeros(self.num_classes, dtype=np.float32), 
                'T':np.zeros(self.num_classes, dtype=np.float32), 
                'TP':np.zeros(self.num_classes, dtype=np.float32)
            } for th in self.thresholds}

class Evaluator_For_mIoU:
    def __init__(self, class_names, ignore_index=255):
        self.class_names = class_names
        self.num_classes = len(self.class_names)

        self.ignore_index = ignore_index

        self.clear()

    def clear(self):
        self.TP = []
        self.P = []
        self.T = []
        
        for _ in range(self.classes):
            self.TP.append(0)
            self.P.append(0)
            self.T.append(0)

    def get_data(self, pred_mask, gt_mask):
        obj_mask = gt_mask<self.ignore_index
        correct_mask = (pred_mask==gt_mask) * obj_mask
        
        P_list, T_list, TP_list = [], [], []
        for i in range(self.num_classes):
            P_list.append(np.sum((pred_mask==i)*obj_mask))
            T_list.append(np.sum((gt_mask==i)*obj_mask))
            TP_list.append(np.sum((gt_mask==i)*correct_mask))

        return (P_list, T_list, TP_list)

    def add_using_data(self, data):
        P_list, T_list, TP_list = data
        for i in range(self.num_classes):
            self.P[i] += P_list[i]
            self.T[i] += T_list[i]
            self.TP[i] += TP_list[i]

    def add(self, pred_mask, gt_mask):
        obj_mask = gt_mask<self.ignore_index
        correct_mask = (pred_mask==gt_mask) * obj_mask

        for i in range(self.num_classes):
            self.P[i] += np.sum((pred_mask==i)*obj_mask)
            self.T[i] += np.sum((gt_mask==i)*obj_mask)
            self.TP[i] += np.sum((gt_mask==i)*correct_mask)

    def get(self, detail=False, clear=True):
        IoU_dic = {}
        IoU_list = []

        FP_list = [] # over activation
        FN_list = [] # under activation

        for i in range(self.num_classes):
            IoU = self.TP[i]/(self.T[i]+self.P[i]-self.TP[i]+1e-10) * 100
            FP = (self.P[i]-self.TP[i])/(self.T[i] + self.P[i] - self.TP[i] + 1e-10)
            FN = (self.T[i]-self.TP[i])/(self.T[i] + self.P[i] - self.TP[i] + 1e-10)

            IoU_dic[self.class_names[i]] = IoU

            IoU_list.append(IoU)
            FP_list.append(FP)
            FN_list.append(FN)
        
        mIoU = np.mean(np.asarray(IoU_list))
        mIoU_foreground = np.mean(np.asarray(IoU_list)[1:])

        FP = np.mean(np.asarray(FP_list))
        FN = np.mean(np.asarray(FN_list))
        
        if clear:
            self.clear()
        
        if detail:
            return mIoU, mIoU_foreground, IoU_dic, FP, FN
        else:
            return mIoU, mIoU_foreground

    