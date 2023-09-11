import numpy as np
from torch.autograd import Variable
import time
from utils import OsJoin
from utils import AverageMeter, calculate_accuracy, calculate_recall
import torch
from torch.utils.data import DataLoader
from opts import parse_opts
from model import generate_model
from dataset import TestSet
from utils import Logger
from torch import nn
import scipy.io
from torchcam.methods import ScoreCAM
opt = parse_opts()

def test_epoch(epoch, data_loader, model, criterion, opt, logger):
    print('test at epoch {}'.format(epoch))
    model.eval()
    model_val = model.eval()
    batch_time =AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()
    recalls = AverageMeter()
    precisions = AverageMeter()
    f1s = AverageMeter()
    sensitivitys = AverageMeter()
    specificitys = AverageMeter()
    end_time = time.time()
    labels_arr = torch.empty(4).cuda()
    pred_arr = torch.empty(4, 1).cuda()
    fc_arr = torch.zeros((opt.sample_size1_fc,opt.sample_size2_fc))
    fa_arr = torch.zeros((opt.sample_size1_dti, opt.sample_size2_dti))
    dfc_arr = torch.zeros((opt.sample_size1_fc, opt.sample_size1_fc,opt.sample_duration_dfc))
    alff_arr = torch.zeros((opt.sample_size1_fmri, opt.sample_size2_fmri,opt.sample_duration_fmri))
    for i ,(inputs,labels) in enumerate(data_loader):
        data_time.update(time.time() - end_time)
        labels = list(map(int,labels))
        inputs = (torch.unsqueeze(input, 1) for input in inputs)
        inputs = (input.type(torch.FloatTensor) for input in inputs)
        # inputs = torch.unsqueeze(inputs,1)
        # inputs = inputs.type(torch.FloatTensor)

        if not opt.no_cuda:
            labels = torch.LongTensor(labels).cuda(non_blocking = True)
        with torch.no_grad():
            inputs = (Variable(input) for input in inputs)
            labels = Variable(labels)
            inputs = list(inputs)
            features_dict = ['ALFF','DFC', 'FA', 'FC']
            features_select = opt.features.split('_')
            indexs = (features_dict.index(feature) for feature in features_select)

            inputs_1 = (inputs[index] for index in indexs)
            inputs = [list(inputs_1), labels]
            loss, outputs = model(inputs)
            #module.Resnet.layer4
            # cam = ScoreCAM(model_val, ['module.Resnet.layer4','module.dfc_pyramid.conv_last_2','module.Resnet.layer4','module.Resnet.layer4'], batch_size=2,
            #        input_shape=[[inputs[0][0].shape[1:],inputs[0][1].shape[1:],inputs[0][2].shape[1:],inputs[0][3].shape[1:]],inputs[1].shape])
            # cam = ScoreCAM(model_val, ['module.Resnet.layer4','module.dfc_pyramid.conv_last_2','module.Resnet.layer4','module.Resnet.layer4'], batch_size=2,
            #        input_shape=[[inputs[0][0].shape[1:],inputs[0][1].shape[1:],inputs[0][2].shape[1:],inputs[0][3].shape[1:]],inputs[1].shape])
            cam = ScoreCAM(model_val, ['module.Resnet.layer4','module.dfc_pyramid.dfc_encoder_40','module.Resnet.layer4',
                                       'module.Resnet.layer4'], batch_size=2,
                   input_shape=[[inputs[0][0].shape[1:],inputs[0][1].shape[1:],inputs[0][2].shape[1:],inputs[0][3].shape[1:]],inputs[1].shape])
            out = model_val(inputs)
            # select the activation map of specific class
            Map = cam(class_idx=1)
            fc_arr = ((Map[2][1,...].squeeze().cpu() + Map[2][0,...].squeeze().cpu())/2+ fc_arr.cpu())/2
            alff_arr = ((Map[0][1, ...].squeeze().cpu() + Map[0][0, ...].squeeze().cpu())/2 + alff_arr.cpu())/2
            dfc_arr = ((Map[1][1, ...].squeeze().cpu() + Map[1][0, ...].squeeze().cpu())/2 + dfc_arr.cpu())/2
            fa_arr =  ((Map[3][1, ...].squeeze().cpu() + Map[3][0, ...].squeeze().cpu())/2 + fa_arr.cpu())/2
            # scipy.io.savemat('Fc_whole.mat', {'test': fc_arr})
            '''
            save the activation map in multi-modal MRI
            '''
            # np.save(r'.\alff_whole.npy',alff_arr)
            np.save(r'.\dfc40_whole.npy', dfc_arr)
            # np.save(r'.\fc_whole.npy', fc_arr)
            # np.save(r'.\fa_whole.npy', fa_arr)
            # inputs = Variable(inputs)
            # outputs = model(inputs)
            #loss = criterion(outputs, labels)
            acc = calculate_accuracy(outputs, labels)
            recall, precision, f1, sensitivity, specificity = calculate_recall(outputs, labels, opt)

        losses.update(loss.data, inputs[0][0].size(0))
        accuracies.update(acc, inputs[0][0].size(0))
        recalls.update(recall, inputs[0][0].size(0))
        precisions.update(precision, inputs[0][0].size(0))
        f1s.update(f1, inputs[0][0].size(0))
        sensitivitys.update(sensitivity, inputs[0][0].size(0))
        specificitys.update(specificity, inputs[0][0].size(0))
        batch_time.update(time.time() - end_time)
        end_time = time.time()

        print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Acc {acc.val:.3f} ({acc.avg:.3f})\t'
              'Recall {recall.val:.3f} ({recall.avg:.3f})\t'
              'Precision {precision.val:.3f} ({precision.avg:.3f})\t'
              'f1 {f1.val:.3f} ({f1.avg:.3f})\t'
              'sensitivity {sensitivity.val:.3f} ({sensitivity.avg:.3f})\t'
              'specificity {specificity.val:.3f} ({specificity.avg:.3f})\t'
        .format(
            epoch,
            i + 1,
            len(data_loader),
            batch_time=batch_time,
            data_time=data_time,
            loss=losses,
            acc=accuracies,
            recall=recalls,
            precision=precisions,
            f1=f1s,
            sensitivity=sensitivitys,
            specificity=specificitys))
        _, pred = outputs.topk(k=1, dim=1, largest=True)
        pred_arr = torch.cat([pred_arr, pred], dim=0)
        labels_arr = torch.cat([labels_arr, labels], dim=0)
    print('prediction :', end=' ')
    for i in range(4, len(pred_arr)):
        print('%d\t'%(pred_arr[i]), end='')
    print('\nlabel    :', end=' ')
    for i in range(4, len(labels_arr)):
        print('%d\t'%(labels_arr[i]), end='')
    print('\n')
    labels_arr = torch.empty(4).cuda()
    pred_arr = torch.empty(4, 1).cuda()
    logger.log({'epoch': epoch, 'loss': round(losses.avg.item(), 4),
                'acc': round(accuracies.avg.item(), 4), 'recall': round(recalls.avg.item(), 4),
                'precision': round(precisions.avg.item(), 4), 'f1': round(f1s.avg.item(), 4),
                'sensitivity': round(sensitivitys.avg.item(), 4), 'specificity': round(specificitys.avg.item(), 4)})
if opt.test:
    if opt.resume_path:
        opt.resume_path = OsJoin(opt.root_path, opt.resume_path)
    test_data = TestSet()
    test_loader = torch.utils.data.DataLoader(test_data, batch_size = opt.batch_size, shuffle=False,
                                                        num_workers = 0, pin_memory=True)
    model, parameters = generate_model(opt)
    criterion = nn.CrossEntropyLoss()
    if not opt.no_cuda:
        criterion = criterion.cuda()
    #log_path = OsJoin(opt.result_path, opt.data_type, opt.model_name + str(opt.model_depth))
    log_path = OsJoin(opt.result_path, opt.data_type, opt.model_name + '_' + str(opt.model_depth),
                      'logs_%s_fold%s_%s_epoch%d' % (opt.category, opt.fold_id, opt.features, opt.n_epochs))
    test_logger = Logger(
        OsJoin(log_path, 'test.log'), ['epoch', 'loss', 'acc', 'recall'])
    print('loading checkpoint{}'.format(opt.resume_path))
    checkpoint = torch.load(opt.resume_path)
    opt.arch = '{}-{}'.format(opt.model_name, opt.model_depth)
    assert opt.arch == checkpoint['arch']

    model.load_state_dict(checkpoint['state_dict'])

    test_epoch(1, test_loader, model, criterion, opt, test_logger)

# def calculate_test_results(output_buffer,sample_id,test_results,labels):
#     outputs =torch.stack(output_buffer)
#     average_score = torch.mean(outputs,dim=0)
#     sorted_scores,locs = torch.topk(average_score,k=1)
#     results=[]
#     for i in range(sorted_scores.size(0)):
#         score = copy.deepcopy(sorted_scores[i])
#         if isinstance(score, torch.Tensor):
#             score = score.data.cpu().numpy()
#             score = score.item()
#         results.append({
#             'label':labels[i],
#             'score':score
#         })
#     test_results['results'][sample_id] = results
#
# def test(data_loader, model, opt, labels):
#     print('test')
#
#     model.eval()
#
#     batch_time = AverageMeter()
#     data_time = AverageMeter()
#
#     end_time = time.time()
#     output_buffer = []
#     sample_id = ''
#     test_results = {'results': {}}
#     with torch.no_grad():
#         for i, (inputs, targets) in enumerate(data_loader):
#             data_time.update(time.time() - end_time)
#
#             inputs = torch.unsqueeze(inputs, 1)  # 在 1 的位置加一个维度
#             inputs = Variable(inputs)
#             outputs = model(inputs)
#             # if not opt.no_softmax_in_test:
#             #outputs = F.softmax(outputs)
#
#             for j in range(outputs.size(0)):
#                 if not (i == 0 and j == 0):
#                     calculate_test_results(output_buffer, sample_id, test_results, labels)
#                     output_buffer = []
#                 output_buffer.append(outputs[j].data.cpu())
#                 sample_id = labels[j]
#             if (i % 100) == 0:
#                 with open(
#                         OsJoin(opt.result_path, '{}.json'.format(
#                             opt.test_subset)), 'w') as f:
#                     json.dump(test_results, f)
#
#             batch_time.update(time.time() - end_time)
#             end_time = time.time()
#
#             print('[{}/{}]\t'
#                   'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
#                   'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'.format(
#                       i + 1,
#                       len(data_loader),
#                       batch_time=batch_time,
#                       data_time=data_time))
#     with open(
#             OsJoin(opt.result_path, opt.data_type, opt.model_name, str(opt.model_depth), '{}.json'.format(opt.test_subset)),
#             'w') as f:
#         json.dump(test_results, f)
