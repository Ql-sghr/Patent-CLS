import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torchvision.transforms import Compose, CenterCrop, Normalize, ToTensor, ToPILImage
from torch.optim.lr_scheduler import LambdaLR, StepLR
import numpy as np
from tqdm import tqdm
from dataset import BatchData
#from demo import model
from cifar import Cifar100
from exam import Exemplar
from copy import deepcopy
from model import TextCNN
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
print(torch.cuda.is_available())
def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    #print("1",mask)  (128,65) 目标类为TRUE,非目标类为False
    return mask


def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    #print("2",mask)  (128,65) 非目标类为TRUE,目标类为False
    return mask


def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(1, keepdims=True)
    #print("t1+t2",t1+t2)
    rt = torch.cat([t1, t2], dim=1)
    return rt
def get_pft(t, mask2):
    t2 = (t * mask2).sum(1, keepdims=True)
    #print("t1+t2",t2)
    return t2
def normalize(logit):
    mean = logit.mean(dim=-1, keepdims=True)
    stdv = logit.std(dim=-1, keepdims=True)
    return (logit - mean) / (1e-7 + stdv)
def dynamic_temperature_schedule(step, total_steps):
    initial_temp = 10.0
    final_temp = 5.0
    temp = final_temp + 0.5 * (initial_temp - final_temp) * (1 + np.cos(np.pi * step / total_steps))
    return temp
class Trainer:
    def __init__(self, total_cls):
        self.total_cls = total_cls
        self.seen_cls = 0
        self.dataset = Cifar100()   #数据集
        self.model=TextCNN(768,100,[3, 4, 5],65)
        print(self.model)
        self.model = nn.DataParallel(self.model, device_ids=[0]).cuda()


        self.input_transform= Compose([
                                ToTensor(),
                               ])
        self.input_transform_eval= Compose([
                                ToTensor(),
        ])


    def test(self, testdata):
        print("test data number : ",len(testdata))
        self.model.eval()
        count = 0
        correct = 0
        wrong = 0
        for i, (image, label) in enumerate(testdata):
            image = image.cuda()
            label = label.view(-1).cuda()
            self.model = self.model.cuda()
            p = self.model(image)
            pred = p[:,:self.seen_cls].argmax(dim=-1)
            correct += sum(pred == label).item()
            wrong += sum(pred != label).item()
        acc = correct / (wrong + correct)
        print("Test Acc: {}".format(acc * 100))
        self.model.train()
        print("---------------------------------------------")
        return acc

    def test1(self, testdata):
        print("test data number : ",len(testdata))
        self.model.eval()
        count = 0
        correct = 0
        wrong = 0
        all_pred = []
        all_label=[]
        for i, (image, label) in enumerate(testdata):
            image = image.cuda()
            label = label.view(-1).cuda()
            p = self.model(image)
            all_label.append(label)
            pred = p[:,:self.seen_cls].argmax(dim=-1)
            all_pred.append(pred)
            correct += sum(pred == label).item()
            wrong += sum(pred != label).item()
        stacked_pred = torch.cat(all_pred, dim=0)
        stacked_label = torch.cat(all_label, dim=0)
        if self.seen_cls == 65:
            # 将 Tensor 转换为 NumPy 数组
            stacked_pred = stacked_pred.cpu().detach().numpy()
            stacked_label = stacked_label.cpu().numpy()

            # 计算混淆矩阵
            conf_matrix = confusion_matrix(stacked_label, stacked_pred)
            np.save("C:/Users/Administrator/Desktop/hunxiao/ce+kd",conf_matrix)
            """
            # 准确度
            accuracy = accuracy_score(stacked_label, stacked_pred)
            print("\nAccuracy:", accuracy)

            # 精确度
            precision = precision_score(stacked_label,stacked_pred, average='weighted')
            print("\nPrecision:", precision)

            # 召回率
            recall = recall_score(stacked_label, stacked_pred, average='weighted')
            print("\nRecall:", recall)

            # F1 分数
            f1 = f1_score(stacked_label, stacked_pred, average='weighted')
            print("\nF1 Score:", f1)

            # 分类报告
            class_report = classification_report(stacked_label,stacked_pred)
            print("\nClassification Report:")
            print(class_report)
            """
        acc = correct / (wrong + correct)
        print("Test Acc: {}".format(acc*100))
        self.model.train()
        print("---------------------------------------------")
        return acc


    def eval(self, criterion, evaldata):
        self.model.eval()
        losses = []
        correct = 0
        wrong = 0
        for i, (image, label) in enumerate(evaldata):
            image = image.cuda()
            label = label.view(-1).cuda()
            p = self.model(image)
            loss = criterion(p, label)
            losses.append(loss.item())
            pred = p[:,:self.seen_cls].argmax(dim=-1)
            correct += sum(pred == label).item()
            wrong += sum(pred != label).item()
        print("Validation Loss: {}".format(np.mean(losses)))
        print("Validation Acc: {}".format(100*correct/(correct+wrong)))
        self.model.train()
        return



    def get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def train(self, batch_size, epoches, lr, max_size,c,d):
        total_cls = self.total_cls
        criterion = nn.CrossEntropyLoss()
        exemplar = Exemplar(max_size, total_cls)
        previous_model = None
        a=c
        b=d
        dataset = self.dataset
        print(dataset)
        test_xs = []
        test_ys = []
        train_xs = []
        train_ys = []

        test_accs = []
        last_acc = []
        for inc_i in range(dataset.batch_num):
            print(f"Incremental num : {inc_i}")
            train, val, test = dataset.getNextClasses(inc_i)
            train_x, train_y = zip(*train)
            test_x, test_y = zip(*test)
            test_xs.extend(test_x)
            test_ys.extend(test_y)
            train_xs, train_ys = exemplar.get_exemplar_train()
            train_xs.extend(train_x)
            train_ys.extend(train_y)
            print(len(train_xs))
            print(len(test_xs))
            train_data = DataLoader(BatchData(train_xs, train_ys, self.input_transform),  #混在一起的训练数据，包含新类别数据也包含旧类别数据
                        batch_size=batch_size, shuffle=True, drop_last=True)
            test_data = DataLoader(BatchData(test_xs, test_ys, self.input_transform_eval),
                        batch_size=batch_size, shuffle=True)
            print("test data number : ", len(test_data))
            optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9,  weight_decay=2e-4)
            scheduler = MultiStepLR(optimizer, [50,100, 150], gamma=0.1)
            exemplar.update(total_cls // dataset.batch_num, (train_x, train_y))
            self.seen_cls = exemplar.get_cur_cls()
            print("seen cls number : ", self.seen_cls)
            test_acc = []

            for epoch in range(epoches):
                print("---"*50)
                print("Epoch", epoch)
                scheduler.step()
                cur_lr = self.get_lr(optimizer)
                print("Current Learning Rate : ", cur_lr)
                self.model.train()
                if inc_i > 0:
                    self.stage1_DKD(train_data, criterion, optimizer,inc_i,a,b)
                    #self.stage1_distill(train_data, criterion, optimizer,inc_i,a,b)
                    #self.stage1(train_data, criterion, optimizer)

                else:
                    self.stage1(train_data, criterion, optimizer)
                acc = self.test(test_data)
                if epoch == epoches-1:
                    last_acc.append(acc)
            if True:
                # Maintaining Fairness
                if inc_i >= 1:
                    self.model.module.weight_align(inc_i)


            self.previous_model = deepcopy(self.model)
            if inc_i == 0:
                # 假设 self.model 是被 DataParallel 包装的模型
                original_model = self.model.module
                torch.save(original_model.state_dict(), 'models/model3.pth')
            acc = self.test(test_data)
            test_acc.append(acc)
            test_accs.append(max(test_acc))
            print(test_accs)
            file_name = r"C:\Users\Administrator\Desktop\lulu.txt"
            if(inc_i==5):

                # 打开文件并写入数据
                with open(file_name, "a") as file:
                    file.write(f"{a}\t{b}\t")
                    file.write(f"\n")
                    file.write("WA前：")
                    for item in last_acc:
                        file.write(f"{item}    ")
                    file.write(f"\n")
                    file.write("WA后：")
                    for item in test_accs:
                        file.write(f"{item}    ")  # 写入每个元素并换行
                    file.write(f"\n")



    def stage1_distill(self, train_data, criterion, optimizer,inc_i,a,b):
        print("Training ... ")
        distill_losses = []
        ce_losses = []
        T = 8
        alpha = (self.seen_cls - 13)/ self.seen_cls
        print("classification proportion 1-alpha = ", 1-alpha)
        for i, (image, label) in enumerate(tqdm(train_data)):
            image = image.cuda()
            label = label.view(-1).cuda()
            self.model = self.model.cuda()
            p = self.model(image)
            with torch.no_grad():
                pre_p = self.previous_model(image)
                pre_p = F.softmax(pre_p[:, :self.seen_cls-13]/T, dim=1)  #pre_p[:,:self.seen_cls-13]
            logp = F.log_softmax(p[:, :self.seen_cls-13]/T, dim=1)     #p[:,:self.seen_cls-13]
            loss_soft_target = -torch.mean(torch.sum(pre_p * logp, dim=1))  #蒸馏损失
            #loss_soft_target = F.kl_div(logp, pre_p, reduction="none").sum(1).mean()
            loss_hard_target = nn.CrossEntropyLoss()(p[:, :self.seen_cls], label)
            #loss_hard_target = criterion(p[:, :self.seen_cls], label)
            loss = loss_soft_target*T*T*alpha + loss_hard_target*(1-alpha)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            distill_losses.append(loss_soft_target.item())
            ce_losses.append(loss_hard_target.item())
        print("stage1 distill loss :", np.mean(distill_losses), "ce loss :", np.mean(ce_losses))


    def stage1(self, train_data, criterion, optimizer):
        print("Training ... ")
        losses = []
        for i, (image, label) in enumerate(tqdm(train_data)):
            image = image.cuda()
            label = label.view(-1).cuda()
            self.model=self.model.cuda()
            p = self.model(image)
            label = label.to(torch.int64)
            loss = criterion(p[:,:self.seen_cls], label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print("stage1 loss :", np.mean(losses))

    def stage1_SKD(self, train_data, criterion, optimizer):
        print("Training ...11111111111111111111111111111111111111 ")
        distill_losses = []
        ce_losses = []
        T =1
        alpha = (self.seen_cls - 13)/ self.seen_cls
        print("classification proportion 1-alpha = ", 1-alpha)
        for i, (image, label) in enumerate(tqdm(train_data)):
            image = image.cuda()
            label = label.view(-1).cuda()
            self.model = self.model.cuda()
            p = self.model(image)
            with torch.no_grad():
                pre_p = self.previous_model(image)
            tea_std = torch.std(pre_p, dim=-1, keepdim=True)  # 旧的标准差
            stu_std = torch.std(p, dim=-1, keepdim=True)  # 新的标准差
            p_s = F.log_softmax(p / stu_std * tea_std / T, dim=1)
            p_t = F.softmax(pre_p/ T, dim=1)
            loss_kd = torch.sum(torch.sum(F.kl_div(p_s, p_t, reduction='none'), dim=-1) * (T * T * torch.ones(p.shape[0], 1).cuda())) / p.shape[0] / p.shape[0] * 0.3
            p = p/ stu_std * tea_std
            loss_softmax = criterion(p, label) * 0.7

            loss = loss_kd + loss_softmax
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            distill_losses.append(loss_softmax.item())
            ce_losses.append(loss_kd.item())
        print("stage1 distill loss :", np.mean(distill_losses), "ce loss :", np.mean(ce_losses))


    def stage1_DKD(self, train_data, criterion, optimizer,inc_i,a,b):
        print("Training ...11111111111111111111111111111111111111 ")
        distill_losses = []
        ce_losses = []
        temperature = 8# 4
        print("inc",inc_i)  # 1 2 3 4
        print("t",temperature)
        print("self.seen_cls", self.seen_cls)
        #alpha = (self.seen_cls - 13) / self.seen_cls
        alpha = (inc_i+2)*0.1
        print("a",alpha)
        print("b",1 - alpha)
        for i, (image, label) in enumerate(tqdm(train_data)):
            image = image.cuda()
            label = label.view(-1).cuda()
            self.model = self.model.cuda()
            p = self.model(image)
            with torch.no_grad():
                pre_p = self.previous_model(image)
            logits_student = p
            logits_teacher = pre_p
            gt_mask = _get_gt_mask(logits_student, label)
            other_mask = _get_other_mask(logits_student, label)
            pred_student = F.softmax(logits_student / temperature, dim=1)
            pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
            pred_student = cat_mask(pred_student, gt_mask, other_mask)
            pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
            log_pred_student = torch.log(pred_student)
            tckd_loss = (
                    F.kl_div(log_pred_student, pred_teacher, size_average=False)
                    * (temperature ** 2)
                    / label.shape[0]

            )
            pred_teacher_part2 = F.softmax(
                logits_teacher / temperature - 1000.0 * gt_mask, dim=1
            )
            log_pred_student_part2 = F.log_softmax(
                logits_student / temperature - 1000.0 * gt_mask, dim=1
            )
            nckd_loss = (
                    F.kl_div(log_pred_student_part2, pred_teacher_part2, size_average=False)
                    * (temperature ** 2)
                    / label.shape[0]
                    # -torch.mean(torch.sum(pred_teacher_part2 * log_pred_student_part2, dim=1))  / label.shape[0] 就是算平均的意思
            )
            #loss_KD = tckd_loss * 0.5 + nckd_loss * 0.5
            loss_KD = tckd_loss * a + nckd_loss * (b+0.5*(inc_i-1))
            loss_hard_target = nn.CrossEntropyLoss()(p[:, :self.seen_cls], label)  # 分类损失
            loss = loss_KD*alpha+loss_hard_target*(1-alpha)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            distill_losses.append(loss_KD.item())
            ce_losses.append(loss_hard_target.item())
        print("stage1 distill loss :", np.mean(distill_losses), "ce loss :", np.mean(ce_losses))



