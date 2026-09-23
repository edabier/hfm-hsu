import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils import losses, utils, extractor

class UnmixingFromFeatures(nn.Module):
    def __init__(self, D, B, c, H=224, alpha=None, upsample=True):
        super(UnmixingFromFeatures, self).__init__()
        self.D = D
        self.alpha = alpha
        self.B = B
        self.c = c
        self.H = H
        self.upsample = upsample

        # Upsampling features
        if self.upsample:
            self.upsampler = nn.Linear(self.alpha**2, self.H**2, bias=False)
        
        # Unmixing network
        self.abundance_estimator = Abundance_estimator(self.D, self.c, kernel_size=1)
        self.decoder = Decoder(B=B, c=c)

    @staticmethod
    def weights_init(m):
        if type(m) == nn.Conv2d:
            nn.init.kaiming_normal_(m.weight.data)

    @staticmethod
    def loss(Y_gt, Y_hat, A_hat, W_sad=1, W_ab=0.6, W_mse=0.09):
        sad = losses.SADLoss()
        mse = nn.MSELoss(reduction='sum')
        
        loss_sad = W_sad * sad(Y_gt, Y_hat)
        loss_ab = W_ab * torch.sqrt(A_hat).mean()

        loss_mse = W_mse * mse(Y_gt, Y_hat)/(torch.norm(Y_gt)**2)
        loss = loss_sad + loss_ab + loss_mse

        return loss

    def get_abundances(self, features):
        
        if self.upsample:
            features_up = self.upsampler(features)
        features_up = features_up.view(
            1, self.D, self.H, self.H
        )

        features_up = utils.standardise(features_up)
        A_hat = self.abundance_estimator(features_up)
        return A_hat
    
    def get_endmembers(self):
        return self.decoder.get_endmembers()
    
    def forward(self, features, Y):
        A_hat = self.get_abundances(features)
        Y_hat = self.decoder(A_hat)
        E_hat = self.decoder.get_endmembers()

        return E_hat, A_hat, Y_hat
 
class Abundance_estimator(nn.Module):
    def __init__(self, D, c, kernel_size=1, scale=1):
        super(Abundance_estimator, self).__init__()

        self.abundance_estimator = nn.Sequential(
            nn.Conv2d(D, c, kernel_size=kernel_size, bias=False, padding="same"),
            nn.LeakyReLU(0.02),
            nn.BatchNorm2d(c),
            nn.Dropout(0.2)
        )        
        self.scale = scale
            
    def forward(self, up_feat):
        A_hat = self.abundance_estimator(up_feat)
        A_hat = F.softmax(self.scale * A_hat, dim=1)
        return A_hat

class Decoder(nn.Module):
    def __init__(self, c, B, kernel_size=1):
        super(Decoder, self).__init__()
        self.B = B
        self.c = c

        padding = kernel_size //2
        self.decoder = nn.Conv2d(in_channels=c, out_channels=B,
                                kernel_size=kernel_size,stride=1,
                                padding=padding, bias=False)

    def forward(self, code):
        code = self.decoder(code)
        return code

    def get_endmembers(self):
        return self.decoder.weight.data.squeeze([2, 3])
   
def init_decoder_weights(model, Y, c, E_est=None, is_unmixer=False, use_sivm=True, normalise=False):
    """
    Initializes the model's decoder weights with VCA or SiVM extracted endmembers, or E_est if provided
    input Y must be of shape (B, N) or (B, H, W) -> no batch
    """

    if E_est != None:
        init_em = E_est
    else:
        if use_sivm:
            init_em = extractor.SiVM(Y, c)
        else:
            init_em = extractor.VCA(Y, c)

    if normalise:
        init_em = utils.normalise(init_em, is_endmember=True)

    model_dict = model.decoder.state_dict()
    model_dict["decoder.weight"][:,:,0,0] = init_em
    model.decoder.load_state_dict(model_dict)
    return model