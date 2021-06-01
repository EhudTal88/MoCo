from DataModule import DataModule
from Model import MoCo,DownStreamTaskModel
import torch
from Trainer import MoCoTrainer
import torchvision.models as models


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(f'Working with {device}, number of GPUs: {torch.cuda.device_count()}')

def model_pipeline():
    '''Phase A - train Momentum Encoder'''

    #HyperParameters
    backbone= models.resnet50(pretrained= True)
    tau= 0.07 # Contrastive loss temperament parameter
    momentum= 0.999 # Momentum-encoder - momentum
    lr= 2e-3 # Learning rate
    wd= 1e-4 # Weight decay parameter
    queue_size= 4096 # Dictionary queue length (number of example entries)
    batch_size=128
    epochs= 200
    image_size = 224 # SxS
    ks = (int(0.1 * image_size) // 2) * 2 + 1  # Gaussian kernel (for blurring).
    # Imagenet database statistics (for data normalization):
    __imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]}

    #Initialize dictionary queue:
    queue= torch.zeros(1000,queue_size).to(device=device) #TODO:update

    # Initialize model and dataloaders:
    dl_train,dl_val,transforms= DataModule(batch_size= batch_size,ks=ks,imagenet_stats=__imagenet_stats)
    moco_model= MoCo(backbone= backbone,transforms=transforms).to(device=device)
    optimizer= torch.optim.SGD(params=moco_model.parameters(),lr=lr,weight_decay=wd)
    # Decrease lr by half @ specified epoch-number milestones:
    lr_schedualer = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[120, 160], gamma=0.5)
    # Define loss function for contrastive loss implementation:
    loss_fn= torch.nn.CrossEntropyLoss()

    # Pre-defined trainer for MoCo scheme:
    trainer= MoCoTrainer(model= moco_model, loss_fn=loss_fn,optimizer=optimizer,scheduler=lr_schedualer,tau=tau,
                        queue=queue,momento=momentum,flg=True,device=device)

    trainer.fit(dl_train= dl_train,dl_val= dl_val, epochs= epochs)

    '''Phase B - train Downstream task'''


if __name__ == '__main__':

    model_pipeline()