import torch
from torch import  nn

class MoCo(nn.Module):
    def __init__(self,backbone,transforms):
        super(MoCo,self).__init__()

        self.keys_encoder= backbone #todo: change backbone
        self.keys_encoder = nn.sequential('MLP',nn.Conv2d(512, 128, (1, 1), stride=(1, 1)))

        #for p in self.keys_encoder.parameters(): #Detach Gradients from keys network
            #p.requires_grad= False

        self.query_encoder= backbone
        self.keys_encoder = nn.sequential('MLP',nn.Conv2d(512, 128, (1, 1), stride=(1, 1)))
        self.classifier= nn.Sequential() #todo: implement MLP Head
        self.transforms= transforms

    def forward(self, x_k,x_q,queue):
        ''' Augment x twice -> pass through k/q networks -> '''
        #forward pass through backbone:

        # Compute encoded representations:
        k= self.keys_encoder(x_k)
        q= self.query_encoder(x_q)
        # Normalize encoded representations:
        k = k/torch.norm(k,2)
        q = q/torch.norm(q,2)

        x_k= x_k.detach()

        N,C= k.shape#shape of input data (NxC)
        _,K= queue.shape

        #Create Logits
        l_pos= torch.bmm(q.view(N,1,C),k.view(N,C,1)).view(N,1) #dim (Nx1)
        l_neg= torch.mm(q.view(N,C),queue.view(C,K)) #dim (NxK)
        logits= torch.cat((l_pos,l_neg),dim=1) #dim (Nx(K+1))


        return logits,k


class DownStreamTaskModel(nn.Module):
    def __init__(self,encoder):
        super(DownStreamTaskModel,self).__init__()

        self.encoder= encoder
        self.fc= nn.Sequential(nn.Linear(),nn.ReLU(),nn.Dropout())

    def forward(self,batch):
        return self.fc(self.encoder(batch))
