import matplotlib.pyplot as plt
import os
import os
from PIL import Image
import numpy as np
import cv2

def loss_plot(args,loss1,loss2):
    num = args.epoch
    x = [i for i in range(num)]
    plot_save_path = r'result/plot/'
    if not os.path.exists(plot_save_path):
        os.makedirs(plot_save_path)
    save_loss = plot_save_path+str(args.arch)+'_loss.png'
    plt.figure()
    plt.plot(x, loss1, label='Train Loss')
    plt.plot(x, loss2, label='Validation Loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.savefig(save_loss)
    plt.show()

def metrics_plot(arg,name,*args):
    num = arg.epoch
    names = name.split('&')
    metrics_value = args
    i=0
    x = [i for i in range(num)]
    plot_save_path = r'result/plot/'
    if not os.path.exists(plot_save_path):
        os.makedirs(plot_save_path)
    save_metrics = plot_save_path + str(arg.arch)+'_'+name+'.png'
    plt.figure()
    for l in metrics_value:
        plt.plot(x,l,label=str(names[i]))
        #plt.scatter(x,l,label=str(l))
        i+=1
    plt.legend()
    plt.savefig(save_metrics)
    
def save_results(inputs, targets, output, count,save_dir):

    input_img = inputs[0][0].cpu().numpy() 
    target_img = targets[0][0].cpu().numpy()  
    output_img = output[0][0].cpu().numpy() 

    input_img = np.uint8(input_img * 255)  
    target_img = np.uint8(target_img * 255)  
    output_img = np.uint8(output_img * 255)  

    # Save images
    input_filename = os.path.join(save_dir, f'input_{count}.png')
    target_filename = os.path.join(save_dir, f'target_{count}.png')
    output_filename = os.path.join(save_dir, f'output_{count}.png')

    cv2.imwrite(input_filename, input_img)
    cv2.imwrite(target_filename, target_img)
    cv2.imwrite(output_filename, output_img)



