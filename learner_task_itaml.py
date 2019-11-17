import os
import torch
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
import torch.optim as optim
import time
import pickle
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb
import copy
from resnet import *
import random
from radam import *


class ResNet_features(nn.Module):
    def __init__(self, original_model):
        super(ResNet_features, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-1])
        
    def forward(self, x):
        x = self.features(x)
        return x
    
class Learner():
    def __init__(self,model,args,trainloader,testloader, use_cuda):
        self.model=model
        self.best_model=model
        self.args=args
        self.title='incremental-learning' + self.args.checkpoint.split("/")[-1]
        self.trainloader=trainloader 
        self.use_cuda=use_cuda
        self.state= {key:value for key, value in self.args.__dict__.items() if not key.startswith('__') and not callable(key)} 
        self.best_acc = 0 
        self.testloader=testloader
        self.test_loss=0.0
        self.test_acc=0.0
        self.train_loss, self.train_acc=0.0,0.0       
        
        meta_parameters = []
        normal_parameters = []
        for n,p in self.model.named_parameters():
            meta_parameters.append(p)
            p.requires_grad = True
            if("fc" in n):
                normal_parameters.append(p)
      
        if(self.args.optimizer=="radam"):
            self.optimizer = RAdam(meta_parameters, lr=self.args.lr, betas=(0.9, 0.999), weight_decay=0)
        elif(self.args.optimizer=="adam"):
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0, amsgrad=False)
        elif(self.args.optimizer=="sgd"):
            self.optimizer = optim.SGD(meta_parameters, lr=self.args.lr, momentum=0.9, weight_decay=0.001)
 

    def learn(self):
        logger = Logger(os.path.join(self.args.checkpoint, 'session_'+str(self.args.sess)+'_log.txt'), title=self.title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.', 'Best Acc'])
            
        for epoch in range(0, self.args.epochs):
            self.adjust_learning_rate(epoch)
            print('\nEpoch: [%d | %d] LR: %f Sess: %d' % (epoch + 1, self.args.epochs, self.state['lr'],self.args.sess))

            self.train(self.model, epoch)
#             if(epoch> self.args.epochs-5):
            self.test(self.model)
        
            # append logger file
            logger.append([self.state['lr'], self.train_loss, self.test_loss, self.train_acc, self.test_acc, self.best_acc])

            # save model
            is_best = self.test_acc > self.best_acc
            if(is_best and epoch>self.args.epochs-10):
                self.best_model = copy.deepcopy(self.model)

            self.best_acc = max(self.test_acc, self.best_acc)
            if(epoch==self.args.epochs-1):
                self.save_checkpoint(self.best_model.state_dict(), True, checkpoint=self.args.savepoint, filename='session_'+str(self.args.sess)+'_model_best.pth.tar')
        self.model = copy.deepcopy(self.best_model)
        
        logger.close()
        logger.plot()
        savefig(os.path.join(self.args.checkpoint, 'log.eps'))

        print('Best acc:')
        print(self.best_acc)
    
    def train(self, model, epoch):
        model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        end = time.time()
        
        bi = self.args.class_per_task*(1+self.args.sess)
        bar = Bar('Processing', max=len(self.trainloader))
        
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            # measure data loading time
            data_time.update(time.time() - end)
            sessions = []
             
            targets_one_hot = torch.FloatTensor(inputs.shape[0], bi)
            targets_one_hot.zero_()
            targets_one_hot.scatter_(1, targets[:,None], 1)
    
            if self.use_cuda:
                inputs, targets_one_hot, targets = inputs.cuda(), targets_one_hot.cuda(),targets.cuda()
            inputs, targets_one_hot, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets_one_hot),torch.autograd.Variable(targets)

            reptile_grads = {}            
            np_targets = targets.detach().cpu().numpy()
            num_updates = 0
            
            outputs2, _ = model(inputs)
            
            model_base = copy.deepcopy(model)
            for task_idx in range(1+self.args.sess):
                idx = np.where((np_targets>= task_idx*self.args.class_per_task) & (np_targets < (task_idx+1)*self.args.class_per_task))[0]
                ai = self.args.class_per_task*task_idx
                bi = self.args.class_per_task*(task_idx+1)
                
                ii = 0
                if(len(idx)>0):
                    sessions.append([task_idx, ii])
                    ii += 1
                    for i,(p,q) in enumerate(zip(model.parameters(), model_base.parameters())):
                        p=copy.deepcopy(q)
                        
                    class_inputs = inputs[idx]
                    class_targets_one_hot= targets_one_hot[idx]
                    class_targets = targets[idx]
                    
                    if(self.args.sess==task_idx and self.args.sess==4 and self.args.dataset=="svhn"):
                        self.args.r = 4
                    else:
                        self.args.r = 1
                        
                    for kr in range(self.args.r):
                        _, class_outputs = model(class_inputs)

                        class_tar_ce=class_targets_one_hot.clone()
                        class_pre_ce=class_outputs.clone()
                        loss = F.binary_cross_entropy_with_logits(class_pre_ce[:, ai:bi], class_tar_ce[:, ai:bi]) 
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()

                    for i,p in enumerate(model.parameters()):
                        if(num_updates==0):
                            reptile_grads[i] = [p.data]
                        else:
                            reptile_grads[i].append(p.data)
                    num_updates += 1
            
            for i,(p,q) in enumerate(zip(model.parameters(), model_base.parameters())):
                alpha = np.exp(-self.args.beta*((1.0*self.args.sess)/self.args.num_task))
#                 alpha = np.exp(-0.05*self.args.sess)
                ll = torch.stack(reptile_grads[i])
#                 if(p.data.size()[0]==10 and p.data.size()[1]==256):
# #                     print(sessions)
#                     for ik in sessions:
# #                         print(ik)
#                         p.data[2*ik[0]:2*(ik[0]+1),:] = ll[ik[1]][2*ik[0]:2*(ik[0]+1),:]*(alpha) + (1-alpha)* q.data[2*ik[0]:2*(ik[0]+1),:]
#                 else:
                p.data = torch.mean(ll,0)*(alpha) + (1-alpha)* q.data  
                    
                
            
        
            # measure accuracy and record loss
            prec1, prec5 = accuracy(output=outputs2.data[:,0:bi], target=targets.cuda().data, topk=(1, 1))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))
            
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix  = '({batch}/{size}) | Total: {total:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f} '.format(
                        batch=batch_idx + 1,
                        size=len(self.trainloader),
                        total=bar.elapsed_td,
                        loss=losses.avg,
                        top1=top1.avg,
                        top5=top5.avg
                        )
            bar.next()
        bar.finish()

        self.train_loss,self.train_acc=losses.avg, top1.avg

    def test(self, model):

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        class_acc = {}
        
        
        # switch to evaluate mode
        model.eval()
        ai = 0
        bi = self.args.class_per_task*(self.args.sess+1)
        
        end = time.time()
        bar = Bar('Processing', max=len(self.testloader))
        for batch_idx, (inputs, targets) in enumerate(self.testloader):
            # measure data loading time
            data_time.update(time.time() - end)
#             print(targets)
            targets_one_hot = torch.FloatTensor(inputs.shape[0], self.args.num_class)
            targets_one_hot.zero_()
            targets_one_hot.scatter_(1, targets[:,None], 1)
            target_set = np.unique(targets)
            
            if self.use_cuda:
                inputs, targets_one_hot,targets = inputs.cuda(), targets_one_hot.cuda(),targets.cuda()
            inputs, targets_one_hot, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets_one_hot) ,torch.autograd.Variable(targets)

            outputs2, outputs = model(inputs)
            loss = F.binary_cross_entropy_with_logits(outputs[ai:bi], targets_one_hot[ai:bi])
                    
            prec1, prec5 = accuracy(outputs2.data[:,0:self.args.class_per_task*(1+self.args.sess)], targets.cuda().data, topk=(1, 1))


            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            pred = torch.argmax(outputs2[:,0:self.args.class_per_task*(1+self.args.sess)], 1, keepdim=False)
            pred = pred.view(1,-1)
            correct = pred.eq(targets.view(1, -1).expand_as(pred)).view(-1) 
            correct_k = float(torch.sum(correct).detach().cpu().numpy())

            for i,p in enumerate(pred.view(-1)):
                key = int(p.detach().cpu().numpy())
                if(correct[i]==1):
                    if(key in class_acc.keys()):
                        class_acc[key] += 1
                    else:
                        class_acc[key] = 1
                        
                        
            # plot progress
            bar.suffix  = '({batch}/{size})  Total: {total:} | Loss: {loss:.4f} | top1: {top1: .4f} | top1_task: {top5: .4f}'.format(
                        batch=batch_idx + 1,
                        size=len(self.testloader),
                        total=bar.elapsed_td,
                        loss=losses.avg,
                        top1=top1.avg,
                        top5=top5.avg
                        )
            bar.next()
        bar.finish()
        self.test_loss= losses.avg;self.test_acc= top1.avg
            
        acc_task = {}
        for i in range(self.args.sess+1):
            acc_task[i] = 0
            for j in range(self.args.class_per_task):
                try:
                    acc_task[i] += class_acc[i*self.args.class_per_task+j]/self.args.sample_per_task_testing[i] * 100
                except:
                    pass
        print("\n".join([str(acc_task[k]).format(".4f") for k in acc_task.keys()]) )    
        print(class_acc)

        
        with open(self.args.savepoint + "/acc_task_test_"+str(self.args.sess)+".pickle", 'wb') as handle:
            pickle.dump(acc_task, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    
    def meta_test(self, model, memory, inc_dataset):

        # switch to evaluate mode
        model.eval()
        
        meta_models = []   
        base_model = copy.deepcopy(model)
        class_acc = {}
        meta_task_test_list = {}
        for task_idx in range(self.args.sess+1):
            
            memory_data, memory_target = memory
            memory_data = np.array(memory_data, dtype="int32")
            memory_target = np.array(memory_target, dtype="int32")
            
            
            mem_idx = np.where((memory_target>= task_idx*self.args.class_per_task) & (memory_target < (task_idx+1)*self.args.class_per_task))[0]
            meta_memory_data = memory_data[mem_idx]
            meta_memory_target = memory_target[mem_idx]
            meta_model = copy.deepcopy(base_model)
            
            meta_loader = inc_dataset.get_custom_loader_idx(meta_memory_data, mode="train", batch_size=64)

            meta_optimizer = optim.Adam(meta_model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0, amsgrad=False)
            
            meta_model.train()
            
            ai = self.args.class_per_task*task_idx
            bi = self.args.class_per_task*(task_idx+1)
            bb = self.args.class_per_task*(self.args.sess+1)
            print("Training meta tasks:\t" , task_idx)
                
            #META training
            if(self.args.sess!=0):
                for ep in range(1):
                    bar = Bar('Processing', max=len(meta_loader))
                    for batch_idx, (inputs, targets) in enumerate(meta_loader):
                        targets_one_hot = torch.FloatTensor(inputs.shape[0], (task_idx+1)*self.args.class_per_task)
                        targets_one_hot.zero_()
                        targets_one_hot.scatter_(1, targets[:,None], 1)
                        target_set = np.unique(targets)

                        if self.use_cuda:
                            inputs, targets_one_hot,targets = inputs.cuda(), targets_one_hot.cuda(),targets.cuda()
                        inputs, targets_one_hot, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets_one_hot) ,torch.autograd.Variable(targets)

                        _, outputs = meta_model(inputs)
                        class_pre_ce=outputs.clone()
                        class_pre_ce = class_pre_ce[:, ai:bi]
                        class_tar_ce=targets_one_hot.clone()

                        loss = F.binary_cross_entropy_with_logits(class_pre_ce, class_tar_ce[:, ai:bi])

                        meta_optimizer.zero_grad()
                        loss.backward()
                        meta_optimizer.step()
                        bar.suffix  = '({batch}/{size})  Total: {total:} | Loss: {loss:.4f}'.format(
                                        batch=batch_idx + 1,
                                        size=len(meta_loader),
                                        total=bar.elapsed_td,
                                        loss=loss)
                        bar.next()
                    bar.finish()

            
            #META testing with given knowledge on task
            meta_model.eval()   
            for cl in range(self.args.class_per_task):
                class_idx = cl + self.args.class_per_task*task_idx
                loader = inc_dataset.get_custom_loader_class([class_idx], mode="test", batch_size=10)

                for batch_idx, (inputs, targets) in enumerate(loader):
                    targets_task = targets-self.args.class_per_task*task_idx

                    if self.use_cuda:
                        inputs, targets_task = inputs.cuda(),targets_task.cuda()
                    inputs, targets_task = torch.autograd.Variable(inputs), torch.autograd.Variable(targets_task)

                    _, outputs = meta_model(inputs)

                    if self.use_cuda:
                        inputs, targets = inputs.cuda(),targets_task.cuda()
                    inputs, targets_task = torch.autograd.Variable(inputs), torch.autograd.Variable(targets_task)

                    pred = torch.argmax(outputs[:,ai:bi], 1, keepdim=False)
                    pred = pred.view(1,-1)
                    correct = pred.eq(targets_task.view(1, -1).expand_as(pred)).view(-1) 

                    correct_k = float(torch.sum(correct).detach().cpu().numpy())

                    for i,p in enumerate(pred.view(-1)):
                        key = int(p.detach().cpu().numpy())
                        key = key + self.args.class_per_task*task_idx
                        if(correct[i]==1):
                            if(key in class_acc.keys()):
                                class_acc[key] += 1
                            else:
                                class_acc[key] = 1
                                

            
#           META testing - no knowledge on task
            meta_model.eval()   
            for batch_idx, (inputs, targets) in enumerate(self.testloader):
                if self.use_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()
                inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
                             
                _, outputs = meta_model(inputs)
                outputs_base, _ = self.model(inputs)
                task_ids = outputs

                task_ids = task_ids.detach().cpu()
                outputs = outputs.detach().cpu()
                outputs = outputs.detach().cpu()
                outputs_base = outputs_base.detach().cpu()
                
                bs = inputs.size()[0]
                for i,t in enumerate(list(range(bs))):
                    j = batch_idx*self.args.test_batch + i
                    output_base_max = []
                    for si in range(self.args.sess+1):
                        sj = outputs_base[i][si* self.args.class_per_task:(si+1)* self.args.class_per_task]
                        sq = torch.max(sj)
                        output_base_max.append(sq)
                    
                    task_argmax = np.argsort(outputs[i][ai:bi])[-5:]
                    task_max = outputs[i][ai:bi][task_argmax]
                    
                    if ( j not in meta_task_test_list.keys()):
                        meta_task_test_list[j] = [[task_argmax,task_max, output_base_max,targets[i]]]
                    else:
                        meta_task_test_list[j].append([task_argmax,task_max, output_base_max,targets[i]])
            del meta_model
                                
        acc_task = {}
        for i in range(self.args.sess+1):
            acc_task[i] = 0
            for j in range(self.args.class_per_task):
                try:
                    acc_task[i] += class_acc[i*self.args.class_per_task+j]/self.args.sample_per_task_testing[i] * 100
                except:
                    pass
        print("\n".join([str(acc_task[k]).format(".4f") for k in acc_task.keys()]) )    
        print(class_acc)
        
        with open(self.args.savepoint + "/meta_task_test_list_"+str(task_idx)+".pickle", 'wb') as handle:
            pickle.dump(meta_task_test_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return acc_task
        

    def get_memory(self, memory, for_memory, seed=1):
        random.seed(seed)
        memory_per_task = self.args.memory // ((self.args.sess+1)*self.args.class_per_task)
        self._data_memory, self._targets_memory = np.array([]), np.array([])
        mu = 1
        
        #update old memory
        if(memory is not None):
            data_memory, targets_memory = memory
            data_memory = np.array(data_memory, dtype="int32")
            targets_memory = np.array(targets_memory, dtype="int32")
            for class_idx in range(self.args.class_per_task*(self.args.sess)):
                idx = np.where(targets_memory==class_idx)[0][:memory_per_task]
                self._data_memory = np.concatenate([self._data_memory, np.tile(data_memory[idx], (mu,))   ])
                self._targets_memory = np.concatenate([self._targets_memory, np.tile(targets_memory[idx], (mu,))    ])
                
                
        #add new classes to the memory
        new_indices, new_targets = for_memory

        new_indices = np.array(new_indices, dtype="int32")
        new_targets = np.array(new_targets, dtype="int32")
        for class_idx in range(self.args.class_per_task*(self.args.sess),self.args.class_per_task*(1+self.args.sess)):
            idx = np.where(new_targets==class_idx)[0][:memory_per_task]
            self._data_memory = np.concatenate([self._data_memory, np.tile(new_indices[idx],(mu,))   ])
            self._targets_memory = np.concatenate([self._targets_memory, np.tile(new_targets[idx],(mu,))    ])
            
        print(len(self._data_memory))
        return list(self._data_memory.astype("int32")), list(self._targets_memory.astype("int32"))

    def save_checkpoint(self, state, is_best, checkpoint, filename):
        if is_best:
            torch.save(state, os.path.join(checkpoint, filename))

    def adjust_learning_rate(self, epoch):
        if epoch in self.args.schedule:
            self.state['lr'] *= self.args.gamma
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.state['lr']
