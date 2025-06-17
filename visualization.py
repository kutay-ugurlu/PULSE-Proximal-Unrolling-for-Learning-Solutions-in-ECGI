import torch 
from matplotlib import pyplot as plt 
from glob import glob 
import os 
import numpy as np 

def visualize_results(result_path):
    
    results = torch.load(result_path)
    fig = plt.figure(figsize=(18,12))

    gs = fig.add_gridspec(3,3)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])
    ax7 = fig.add_subplot(gs[2, 0:3])


    ax1.plot(results['train_loss'], linewidth=2.5)
    ax1.set_title('Train Loss',fontweight='bold')

    ax2.plot(results['train_cc'], linewidth=2.5)
    ax2.set_title('Train CC',fontweight='bold')

    ax3.plot(results['train_re'], linewidth=2.5)
    ax3.set_title('Train RE',fontweight='bold')

    ax4.plot(results['val_loss'], linewidth=2.5)
    ax4.set_title('Validation Loss',fontweight='bold')

    ax5.plot(results['val_cc'], linewidth=2.5)
    ax5.set_title('Validation CC',fontweight='bold')

    ax6.plot(results['val_re'], linewidth=2.5)
    ax6.set_title('Validation RE',fontweight='bold')

    regs = results['reg_container']
    L_reg = regs.shape[1]
    ax7.plot(np.abs(regs), linewidth=2.5)
    plt.legend(['$\lambda_{}$'.format(i+1) for i in range(L_reg)])
    ax7.set_title('Regularization Parameters',fontweight='bold')
    
    model_str = str(results['modelclass']) + "(" + " ,".join([key+":"+str(value) for key,value in results['essential_dict'].items()]) +")"
    fig.suptitle(model_str,fontweight='bold')
    
    for ax in fig.get_axes():
        for label in ax.get_yticklabels():
            label.set_weight("bold")
        for label in ax.get_xticklabels():
            label.set_weight("bold")
    
    return fig

def get_the_latest_file_in_path(path, index=-1):
    list_of_files = glob(path) # * means all if need specific format then *.csv
    sorted_files = sorted(list_of_files, key=os.path.getctime)
    return sorted_files[index]