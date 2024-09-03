from metrics import *
import argparse
import logging
import torch
import matplotlib.pyplot as plt
from torch import autograd, optim
from Loss import *
from dataset import *
from metrics import *
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from plot import *

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

def seed_torch(seed=3407):
    random.seed(seed)
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    
    os.environ['PYTHONHASHSEED'] = str(seed) 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    #torch.use_deterministic_algorithms(True)  # 有检查操作，看下文区别
seed_torch()

def getArgs():
    parse = argparse.ArgumentParser()
    parse.add_argument("--action", type=str, help="train/test/train&test", default="train&test")
    parse.add_argument('--arch', '-a', metavar='ARCH', default='DIEP-Unet',
                       help='DIEP-Unet')
    parse.add_argument("--batch_size", type=int, default=1)
    parse.add_argument('--dataset', default='COVID19_CT_Lung_and_Infection_Segmentation_Dataset',  # dsb2018_256
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
    # model = NestedUNet(1,1).to(device)
    # model = AttU_Net(1,1).to(device)
    model = nCoVSegNet(1).to(device)
    
    return model

def getDataset(args):
    train_dataset = COVID19_CT_Lung_and_Infection_Segmentation_Dataset(r'train', transform=x_transforms, target_transform=y_transforms)
    train_dataloaders = DataLoader(train_dataset, batch_size=args.batch_size)

    val_dataset = COVID19_CT_Lung_and_Infection_Segmentation_Dataset(r"val", transform=x_transforms, target_transform=y_transforms)
    val_dataloaders = DataLoader(val_dataset, batch_size=1)

    test_dataset = COVID19_CT_Lung_and_Infection_Segmentation_Dataset(r"test", transform=x_transforms,target_transform=y_transforms)
    test_dataloaders = DataLoader(test_dataset, batch_size=1)

    return train_dataloaders,val_dataloaders,test_dataloaders

def test(model,test_dataloaders,save_predict=False):
    logging.info('final test........')
    if save_predict ==True:
        dir = os.path.join(r'saved_predict',str(args.arch))
        if not os.path.exists(dir):
            os.makedirs(dir)
        else:
            print('saved_predict already exist!')
    model.load_state_dict(torch.load('save_model/' + args.arch + '.pth'))  # 载入训练好的模型
    

    model.eval()
    with torch.no_grad():
        running_loss = 0.0
        acc = 0.  # Accuracy
        SE = 0.  # Sensitivity (Recall)
        SP = 0.  # Specificity
        PC = 0.  # Precision
        F1 = 0.  # F1 Score
        JS = 0.  # Jaccard Similarity
        DC = 0.  # Dice Coefficient
        HD = 0.
        HD95 = 0.
        count = 0
        
        for inputs, targets in test_dataloaders:
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
            # loss = loss0 + loss1 + loss2 + loss3 + loss4

            # loss = criterion(outputs, targets)
            running_loss += loss.item()
            
            output = outputs.cpu().numpy()
            target = targets.cpu().numpy()
            
            # if(count>=71 and count<=80):
            # if count==240 or count==245  or count==250 or count==280 or count==290 or count==310:
            if count==54 or count==56  or count==69 or count==74 or count==80 or count==310 or count==315 or count==320:
                save_results(inputs, targets, outputs,count,dir)
                
            acc += get_accuracy(output, target)
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
        acc = acc / count
        SE = SE / count
        SP = SP / count
        PC = PC / count
        F1 = F1 / count
        JS = JS / count
        DC = DC / count
        HD = HD / count
        HD95 = HD95 / count

        print('test_Loss: {:.4f},Acc: {:.8f},Sen: {:.8f},Spec: {:.8f},Prec: {:.8f},F1: {:.8f} ,Jaccard: {:.8f} ,Dice: {:.8f},HD: {:.8f},HD95: {:.8f}'
              .format(epoch_loss, acc , SE, SP, PC , F1, JS, DC, HD, HD95))
        logging.info('test_Loss=%f, Acc=%f,Sen=%f,Spec=%f,Prec=%f,F1=%f,Jaccard=%f,Dice=%f,HD=%f,HD95=%f' % (epoch_loss, acc , SE, SP, PC , F1, JS, DC,HD, HD95))
        print('***' * 10+' Test End '+'***' * 10)
        
        
        
        
        
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
    print('***' * 10+' Test Start '+'***' * 10)
    print('---' * 20)
    print('models:%s,\nbatch size:%s\ndataset:%s' % \
          (args.arch, args.batch_size,args.dataset))
    logging.info('\n=======\nmodels:%s,\nbatch size:%s\ndataset:%s\n========' % \
          (args.arch,args.batch_size,args.dataset))
    print('---' * 20)
    model = getModel(args)
    train_dataloaders,val_dataloaders,test_dataloaders = getDataset(args)
    criterion = bce_dice_loss
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    if 'test' in args.action:
        test(model,test_dataloaders, save_predict=True)
        
from models.module.utils import CalParams
x = torch.randn(1, 1, 352, 352).cuda()
CalParams(model, x)