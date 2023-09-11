import torch
from torch.autograd import Variable
import time
from utils import AverageMeter, calculate_accuracy, calculate_recall

def val_epoch(epoch,data_loader,model,criterion,opt,logger, writer):
    print('validation at epoch {}'.format(epoch) )
    model.eval()

    batch_time =AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()
    recalls = AverageMeter()
    precisions = AverageMeter()
    f1s = AverageMeter()
    sensitivitys = AverageMeter()
    specificitys = AverageMeter()
    writer = writer

    end_time = time.time()

    labels_arr = torch.empty(4).cuda()
    pred_arr = torch.empty(4, 1).cuda()
    for i ,(inputs,labels) in enumerate(data_loader):
        data_time.update(time.time() - end_time)
        labels = list(map(int,labels))
        inputs= (torch.unsqueeze(input,1) for input in inputs)
        inputs = (input.type(torch.FloatTensor) for input in inputs)
        #inputs = inputs.type(torch.FloatTensor)

        if not opt.no_cuda:
            labels = torch.LongTensor(labels).cuda(non_blocking=True)#async to non_blocking
        with torch.no_grad():
            #inputs = Variable(inputs)
            inputs = (Variable(input) for input in inputs)
            labels = Variable(labels)
            inputs = list(inputs)
            features_dict = ['ALFF','DFC', 'FA', 'FC']
            features_select = opt.features.split('_')
            indexs = (features_dict.index(feature) for feature in features_select)
            inputs_1 = (inputs[index] for index in indexs)
            inputs = [list(inputs_1), labels]
            loss, outputs = model(inputs)
            acc = calculate_accuracy(outputs, labels)
            recall, precision, f1, sensitivity, specificity = calculate_recall(outputs, labels, opt)

        losses.update(loss.data, inputs[0][0].size(0))
        accuracies.update(acc, inputs[0][0].size(0))
        recalls.update(recall, inputs[0][0].size(0))
        precisions.update(precision, inputs[0][0].size(0))
        f1s.update(f1, inputs[0][0].size(0))
        sensitivitys.update(sensitivity, inputs[0][0].size(0))
        specificitys.update(specificity, inputs[0][0].size(0))
#        aucs.update(auc, inputs[0][0].size(0))
        batch_time.update(time.time() - end_time)
        end_time = time.time()

        print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Acc {acc.val:.3f} ({acc.avg:.3f})\t'
              'Recall {recall.val:.3f} ({recall.avg:.3f})\t'
              'Precision {precision.val:.3f} ({precision.avg:.3f})\t\n'
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
               'precision': round(precisions.avg.item(), 4),  'f1': round(f1s.avg.item(), 4),
               'sensitivity': round(sensitivitys.avg.item(), 4), 'specificity': round(specificitys.avg.item(), 4)} )
    writer.add_scalar('val/loss', losses.avg, epoch)
    writer.add_scalar('val/accuracy', accuracies.avg, epoch)
    writer.add_scalar('val/recall', recalls.avg, epoch)
    writer.add_scalar('val/precision', precisions.avg, epoch)
    writer.add_scalar('val/f1', f1s.avg, epoch)
    writer.add_scalar('val/sensitivity', sensitivitys.avg, epoch)
    writer.add_scalar('val/specificity', specificitys.avg, epoch)
#    writer.add_scalar('val/auc', aucs.avg, epoch)
    return losses.avg
