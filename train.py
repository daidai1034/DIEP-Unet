import argparse
import logging
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch import autograd, optim
from Loss import *
from dataset import *
from metrics import *
from torchvision.transforms import transforms
from plot import loss_plot
from plot import metrics_plot

# from models.UNet import UNet
# from models.NestedUnet import *
# from models.Att_Unet import AttU_Net
# from models.PAttUnet import PAttUNet
# from models.nCoVSegNet import nCoVSegNet
from models.nCoVSegNet import nCoVSegNet
# from models.DIEP-Unet import UNet

# from models.backbone import UNet
# from models.CMFM import UNet
# from models.FIBE import UNet
# from models.MSIE import UNet
# from models.HAGCA import UNet
# from models.DS import UNet


# from models.MSIE_HAGCA_DS import UNet
# from models.FIBE_HAGCA_DS import UNet
# from models.HAGCA_DS import UNet
# from models.FIBE_MSIE_DS import UNet
# from models.FIBE_MSIE_HAGCA import UNet






def getArgs():
    parse = argparse.ArgumentParser()
    parse.add_argument("--action", type=str, help="train/test/train&test", default="train&test")
    parse.add_argument("--epoch", type=int, default=50)
    parse.add_argument('--arch', '-a', metavar='ARCH', default='DIEP-Unet',
                       help='DIEP-Unet')
    parse.add_argument("--batch_size", type=int, default=4)
    parse.add_argument('--dataset', default='COVID19_CT_Lung_and_Infection_Segmentation_Dataset',
                       help='dataset name:COVID19_CT_Lung_and_Infection_Segmentation_Dataset')
    parse.add_argument("--log_dir", default='result/log', help="log dir")
    parse.add_argument("--save_model", default='save_model', help="save model")
    args = parse.parse_args()
    return args

def getLog(args):
    dirname = os.path.join(args.log_dir,args.arch)
    filename = dirname +'/log.log'
    if not os.path.exists(dirname):
        os.makedirs(os.path.dirname(filename))
    logging.basicConfig(
            filename=filename,
            level=logging.DEBUG,
            format='%(asctime)s:%(levelname)s:%(message)s'
        )
    return logging

def getModel(args):
    # model = UNet(1,1).to(device)
    # model = PAttUNet(1,1).to(device)
    model = nCoVSegNet(1).to(device)
    # model = NestedUNet(1,1).to(device)
    # model = AttU_Net(1,1).to(device)


    return model

def getDataset(args):
    train_dataset = COVID19_CT_Lung_and_Infection_Segmentation_Dataset(r'train', transform=x_transforms, target_transform=y_transforms)
    train_dataloaders = DataLoader(train_dataset, batch_size=args.batch_size)

    val_dataset = COVID19_CT_Lung_and_Infection_Segmentation_Dataset(r"val", transform=x_transforms, target_transform=y_transforms)
    val_dataloaders = DataLoader(val_dataset, batch_size=1)

    test_dataset = COVID19_CT_Lung_and_Infection_Segmentation_Dataset(r"test", transform=x_transforms, target_transform=y_transforms)
    test_dataloaders = DataLoader(test_dataset, batch_size=1)

    return train_dataloaders,val_dataloaders,test_dataloaders

def val(model,best_F1score,val_dataloaders):
    model.eval()
    with torch.no_grad():
        running_loss = 0.0
        Acc = 0.  # Accuracy
        SE = 0.  # Sensitivity (Recall)
        SP = 0.  # Specificity
        PC = 0.  # Precision
        F1 = 0.  # F1 Score
        JS = 0.  # Jaccard Similarity
        DC = 0.  # Dice Coefficient
        HD = 0.
        HD95 = 0.
        count = 0
        for inputs, targets in val_dataloaders:
            inputs = inputs.float().cuda()  # (10,1,224,224)
            targets = targets.float().cuda()
            # outputs = model(inputs)
            # outputs = torch.sigmoid(outputs)

            outputs, lateral_map_3, lateral_map_2, lateral_map_1 = model(inputs)
            # outputs, lateral_map_3, lateral_map_2, lateral_map_1,lateral_map_0 = model(inputs)

            outputs = torch.sigmoid(outputs)
            lateral_map_3 = torch.sigmoid(lateral_map_3)
            lateral_map_2 = torch.sigmoid(lateral_map_2)
            lateral_map_1 = torch.sigmoid(lateral_map_1)
            # lateral_map_0 = torch.sigmoid(lateral_map_0)

            loss4 = criterion(outputs, targets)
            loss3 = criterion(lateral_map_3, targets)
            loss2 = criterion(lateral_map_2, targets)
            loss1 = criterion(lateral_map_1, targets)
            # loss0 = criterion(lateral_map_0, targets)
            loss = loss1 + loss2 + loss3 + loss4
#             loss = loss0 + loss1 + loss2 + loss3 + loss4

            # loss = criterion(outputs, targets)
            running_loss += loss.item()

            output = outputs.cpu().numpy()
            target = targets.cpu().numpy()
            # print('output:',np.max(output),np.min(output))
            # print('target:',np.max(target),np.min(target))
            Acc += get_accuracy(output, target)
            SE += get_sensitivity(output, target)
            SP += get_specificity(output, target)
            PC += get_precision(output, target)
            F1 += get_F1(output, target)
            JS += get_JS(output, target)
            DC += get_DC(output, target)
            HD += get_HD(output, target)
            HD95 += get_HD95(output, target)
            count += 1
        epoch_loss = running_loss/count
        Acc = Acc /count
        SE = SE / count
        SP = SP / count
        PC = PC / count
        F1 = F1 / count
        JS = JS / count
        DC = DC / count
        HD = HD / count
        HD95 = HD95 / count
        print('valid_Loss: {:.4f},Acc: {:.8f},Sen: {:.8f},Spec: {:.8f},Prec: {:.8f},F1: {:.8f} ,Jaccard: {:.8f} ,Dice: {:.8f},HD: {:.8f},HD95: {:.8f}'
              .format(epoch_loss, Acc , SE, SP, PC , F1, JS, DC, HD, HD95))
        logging.info('valid_Loss=%f, Acc=%f,Sen=%f,Spec=%f,Prec=%f,F1=%f,Jaccard=%f,Dice=%f,HD=%f,HD95=%f' % (epoch_loss, Acc , SE, SP, PC , F1, JS, DC, HD, HD95))
        if F1 > best_F1score:
            print('F1:{} > best_F1score:{}'.format(F1,best_F1score))
            logging.info('F1:{} > best_F1score:{}'.format(F1,best_F1score))
            logging.info('===========>save best model!')
            best_F1score = F1
            print('===========>save best model!')
            filename = os.path.join(args.save_model, args.arch + '.pth')
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            torch.save(model.state_dict(), filename)
        return  F1, DC,Acc,best_F1score,epoch_loss

def train(model, criterion, optimizer, train_dataloader,val_dataloader, args):
    aver_F1,aver_dice,aver_acc, best_F1score = 0,0,0,0
    num_epochs = args.epoch
    valid_loss_list =[]
    train_loss_list = []
    F1score_list = []
    dice_list = []
    acc_list = []
    for epoch in range(num_epochs):
        model = model.train()
        print('-' * 20)
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        logging.info('Epoch {}/{}'.format(epoch+1, num_epochs))
        running_loss = 0.0
        count = 0
        for inputs, targets in train_dataloader:
            inputs = inputs.to(device)#[30, 1, 224, 224]
            targets = targets.to(device)

            optimizer.zero_grad()

            # outputs = model(inputs)
            # outputs = torch.sigmoid(outputs)

            outputs, lateral_map_3, lateral_map_2, lateral_map_1 = model(inputs)
#             outputs, lateral_map_3, lateral_map_2, lateral_map_1,lateral_map_0 = model(inputs)

            outputs = torch.sigmoid(outputs)
            lateral_map_3 = torch.sigmoid(lateral_map_3)
            lateral_map_2 = torch.sigmoid(lateral_map_2)
            lateral_map_1 = torch.sigmoid(lateral_map_1)
#             lateral_map_0 = torch.sigmoid(lateral_map_0)

            loss4 = criterion(outputs, targets)
            loss3 = criterion(lateral_map_3, targets)
            loss2 = criterion(lateral_map_2, targets)
            loss1 = criterion(lateral_map_1, targets)
#             loss0 = criterion(lateral_map_0, targets)
            loss = loss1 + loss2 + loss3 + loss4
#             loss = loss0 + loss1 + loss2 + loss3 + loss4

            # loss = criterion(outputs,targets)
            count+=1
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_loss = running_loss/count

        print("Epoch{},train_loss: {:.4f}".format(epoch+1, epoch_loss))
        logging.info("epoch %d train_loss:%0.3f" % (epoch+1,  epoch_loss))

        aver_F1,aver_dice,aver_acc,best_F1score,valid_loss = val(model,best_F1score,val_dataloader)

        train_loss_list.append(epoch_loss)
        valid_loss_list.append(valid_loss)
        F1score_list.append(aver_F1.item())
        dice_list.append(aver_dice.item())
        acc_list.append(aver_acc.item())

    loss_plot(args, train_loss_list,valid_loss_list)
    metrics_plot(args, 'F1score', F1score_list)
    metrics_plot(args, 'dice',dice_list)
    metrics_plot(args,'acc',acc_list)
    return model



if __name__ =="__main__":
    x_transforms = transforms.Compose([
        transforms.ToTensor(),  # -> [0,1]
        transforms.Normalize([0.5], [0.5])  # ->[-1,1]
    ])
    # mask只需要转换为tensor
    y_transforms = transforms.ToTensor()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = getArgs()
    logging = getLog(args)
    print('***' * 20)
    print('models:%s,\nepoch:%s,\nbatch size:%s\ndataset:%s' % \
          (args.arch, args.epoch, args.batch_size,args.dataset))
    logging.info('\n=======\nmodels:%s,\nepoch:%s,\nbatch size:%s\ndataset:%s\n========' % \
          (args.arch, args.epoch, args.batch_size,args.dataset))
    print('***' * 20)
    model = getModel(args)
    train_dataloaders,val_dataloaders,test_dataloaders = getDataset(args)
    criterion = bce_dice_loss
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    if 'train' in args.action:
        train(model, criterion, optimizer, train_dataloaders,val_dataloaders, args)