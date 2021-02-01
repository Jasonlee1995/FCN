import torch


def pixel_acc(output, y):
    predict = torch.argmax(output, dim=1)
    n, h, w = predict.shape
    correct = torch.sum(predict==y).item()
    cnt = n*h*w - torch.sum(y==255).item()
    
    return correct/cnt
    

def mIoU(output, y, num_classes, gpu_id):
    predict = torch.argmax(output, dim=1)
    true_positive = torch.zeros(num_classes).cuda(gpu_id, non_blocking=True)
    false_positive = torch.zeros(num_classes).cuda(gpu_id, non_blocking=True)
    false_negative = torch.zeros(num_classes).cuda(gpu_id, non_blocking=True)
    
    filter_255 = y!=255
    
    for i in range(num_classes):
        positive_i = predict==i
        true_i = y==i
        true_positive[i] += torch.sum(positive_i & true_i)
        false_positive[i] += torch.sum(positive_i & ~true_i & filter_255)
        false_negative[i] += torch.sum(~positive_i & true_i)
        
    return true_positive, false_positive, false_negative