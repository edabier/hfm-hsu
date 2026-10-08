import torch.nn as nn
import torch.nn.functional as F
import torch
from functools import partial
import sys
import os
from safetensors.torch import load_file
from collections import OrderedDict

from src.utils import utils

sys.path.insert(0, '..')

# HFMs
# from src.models.OFA import OFAViT
# from src.models.HyperSIGMA import HyperSIGMA_Unmix
# from SpecAware.example import models_SpecAware_encoder # SpecAware
# from spectral_earth.src.backbones import spec_vit, spec_resnet # SpecViT
# from panopticon.dinov2.models import vision_transformer as panopticon_vits # Panopticon
# import hubconf as buildUniverSat # UniverSat
# import HyperFree.HyperFree.build_HyperFree as hf # HyperFree
# from HyperFree.HyperFree.modeling import image_encoder # HyperFree
# from HyperSL.engine.model import SpectralSharedEncoder # HyperSL
# from RAMEN.pretraining.src.models.networks.encoder import RamenViT # RAMEN

def instantiate_fm(fm_name, Y, c=None, wavelengths=None, size="large", version="v1", path="."):
    _, B, H, _ = Y.shape
    device = Y.device

    if fm_name == "DOFA":
        if H < 224:
            Y = F.interpolate(Y, size=(224,224))
            new_H = 224
        else:
            new_H = 224
        Y = Y[:,:,:new_H, :new_H]

        if size == "large":
            out_indices = [23]
            if version == "v2":
                try:
                    check_point = torch.load(f'{path}/pretrained_models/DOFA/dofav2_vit_large_e150.pth', map_location=device)
                except:
                    print(f"No pretrained weights found for {fm_name}, please download them and add them to the corresponding folder")
                check_model = {
                    k[len("model."):]: v
                    for k, v in check_point.items()
                    if k.startswith("model.")
                }
                fm = OFAViT(
                    img_size=224, patch_size=14, embed_dim=1024, depth=24, num_heads=16, out_indices=out_indices, mlp_ratio=4,
                    norm_layer=partial(nn.LayerNorm, eps=1e-6))
                fm.load_state_dict(check_point, strict=False)
                fm.load_state_dict(check_model, strict=False)
            else:
                try:
                    check_point = torch.load(f'{path}/pretrained_models/DOFA/DOFA_ViT_large_e100.pth', map_location=device)
                except:
                    print(f"No pretrained weights found for {fm_name}, please download them and add them to the corresponding folder")
                fm = OFAViT(
                    img_size=224, patch_size=16, embed_dim=1024, depth=24, num_heads=16, out_indices=out_indices, mlp_ratio=4,
                    norm_layer=partial(nn.LayerNorm, eps=1e-6))
                fm.load_state_dict(check_point, strict=False)

        elif size == "base":
            out_indices = [11]
            if version == "v2":
                try:
                    check_point = torch.load(f'{path}/pretrained_models/DOFA/dofav2_vit_base_e150.pth', map_location=device)
                except:
                    print(f"No pretrained weights found for {fm_name}, please download them and add them to the corresponding folder")    
                check_model = {
                    k[len("model."):]: v
                    for k, v in check_point.items()
                    if k.startswith("model.")
                }
                fm = OFAViT(
                    img_size=224, patch_size=14, embed_dim=768, depth=12, num_heads=12, out_indices=out_indices, mlp_ratio=4,
                    norm_layer=partial(nn.LayerNorm, eps=1e-6))
                fm.load_state_dict(check_point, strict=False)
                fm.load_state_dict(check_model, strict=False)
            else:
                try:
                    check_point = torch.load(f'{path}/pretrained_models/DOFA/DOFA_ViT_base_e100.pth', map_location=device)
                except:
                    print(f"No pretrained weights found for {fm_name}, please download them and add them to the corresponding folder")
                fm = OFAViT(
                    img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, out_indices=out_indices, mlp_ratio=4,
                    norm_layer=partial(nn.LayerNorm, eps=1e-6))
                fm.load_state_dict(check_point, strict=False)

    elif fm_name == "HyperFree":
        if H < 1024:
            Y = F.interpolate(Y, size=(1024,1024))
            new_H = 1024
        else:
            new_H = 1024
        Y = Y[:,:,:new_H, :new_H]

        if size == "base":
            try:
                checkpoint_path = f"{path}/pretrained_models/HyperFree/HyperFree-b.pth"
            except:
                    print(f"No pretrained weights found for {fm_name}, please download them and add them to the corresponding folder")
            fm = image_encoder.ImageEncoderViT(depth=12, embed_dim=768,
                    img_size=new_H, mlp_ratio=4, norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
                    num_heads=12, patch_size=16, qkv_bias=True,
                    use_rel_pos=True, global_attn_indexes=[2, 5, 8, 11],
                    merge_indexs = [3, 12], window_size=14, out_chans=256)

        elif size == "large":
            try:
                checkpoint_path = f"{path}/pretrained_models/HyperFree/HyperFree-l.pth"
            except:
                    print(f"No pretrained weights found for {fm_name}, please download them and add them to the corresponding folder")
            fm = image_encoder.ImageEncoderViT(depth=12, embed_dim=1024,
                    img_size=1024, mlp_ratio=4, norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
                    num_heads=16, patch_size=16, qkv_bias=True,
                    use_rel_pos=True, global_attn_indexes=[5, 11, 17, 23],
                    merge_indexs = [6, 24], window_size=14, out_chans=256)

        elif size == "huge":
            try:
                checkpoint_path = f"{path}/pretrained_models/HyperFree/HyperFree-h.pth"
            except:
                print(f"No pretrained weights found for {fm_name}, please download them and add them to the corresponding folder")
            fm = image_encoder.ImageEncoderViT(depth=12, embed_dim=1280,
                    img_size=1024, mlp_ratio=4, norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
                    num_heads=16, patch_size=16, qkv_bias=True,
                    use_rel_pos=True, global_attn_indexes=[7, 15, 23, 31],
                    merge_indexs = [8, 32], window_size=14, out_chans=256)

        fm = hf.load_and_resize_params(fm, checkpoint_path)

    elif fm_name == "SpecAware":
        new_H = 224
        if H < new_H:
            Y = F.interpolate(Y, size=(new_H, new_H))
        Y = Y[:,:,:new_H, :new_H]
        out_indices = [11]
            
        fm = models_SpecAware_encoder.MaskedHSIAutoencoderViT(embed_dim=768, patch_size=8,
                                    depth=12, num_heads=12, mlp_ratio=4, out_indices=out_indices,
                                    norm_layer=partial(nn.LayerNorm, eps=1e-6))
        try:
            checkpoint = torch.load(f"{path}/pretrained_models/SpecAware/SpecAware_Base_model.pth", map_location=device)
        except:
            print(f"No pretrained weights found for {fm_name}, please download them and add them to the corresponding folder")
        fm.load_state_dict(checkpoint)
        fm = fm.to(device)

    elif fm_name == "HyperSIGMA":
        new_H = H

        embed_dim, seg_patches, NUM_TOKENS, scale, patch_size = 768, 2, 64, 1, 64
        fm = HyperSIGMA_Unmix(patch_size=patch_size, channels=B, seg_patches=seg_patches, NUM_TOKENS=NUM_TOKENS, embed_dim=embed_dim, num_em=c, scale=scale)

        try:
            spat_path = f"{path}/pretrained_models/HyperSIGMA/spat-vit-base-ultra-checkpoint-1599.pth"
            spec_path = f"{path}/pretrained_models/HyperSIGMA/spec-vit-base-ultra-checkpoint-1599.pth"
        except:
            print(f"No pretrained weights found for {fm_name}, please download them and add them to the corresponding folder")
        Spat_pernet = torch.load(spat_path, map_location=torch.device('cpu'), weights_only=False)
        Spat_pernet = Spat_pernet['model']
        for k in list(Spat_pernet.keys()):
            if 'patch_embed.proj' in k:
                del Spat_pernet[k]
        for k in list(Spat_pernet.keys()):
            k_ = 'spat_encoder.' + k
            Spat_pernet[k_] = Spat_pernet.pop(k)

        Spec_pernet = torch.load(spec_path, map_location=torch.device('cpu'), weights_only=False)
        Spec_pernet = Spec_pernet['model']
        for k in list(Spec_pernet.keys()):
            if 'spec' in k:
                del Spec_pernet[k]
            if 'spat' in k:
                del Spec_pernet[k]
        for k in list(Spec_pernet.keys()):
            k_ = 'spec_encoder.' + k
            Spec_pernet[k_] = Spec_pernet.pop(k)

        model_params = fm.state_dict()
        same_parsms = {k: v for k, v in Spat_pernet.items() if k in model_params.keys()}
        model_params.update(same_parsms)
        fm.load_state_dict(model_params)

        same_parsms = {k: v for k, v in Spec_pernet.items() if k in model_params.keys()}
        model_params.update(same_parsms)
        fm.load_state_dict(model_params)

    elif fm_name == "SpecViT":
        if H < 128:
            Y = F.interpolate(Y, size=(128,128))
            new_H = 128
        else:
            new_H = 128
        Y = Y[:,:,:new_H, :new_H]

        if size == "small":
            fm = spec_vit.SpecViTSmall()
            try:
                checkpoint = torch.load(f"{path}/pretrained_models/SpectralEarth/spec_ViTs_mae.pth", map_location=device)
            except:
                    print(f"No pretrained weights found for {fm_name}, please download them and add them to the corresponding folder")
                
        elif size == "base":
            fm = spec_vit.SpecViTBase()
            try:
                checkpoint = torch.load(f"{path}/pretrained_models/SpectralEarth/spec_ViTb_mae.pth", map_location=device)
            except:
                    print(f"No pretrained weights found for {fm_name}, please download them and add them to the corresponding folder")

        elif size == "large":
            fm = spec_vit.SpecViTLarge()
            try:
                checkpoint = torch.load(f"{path}/pretrained_models/SpectralEarth/spec_ViTl_mae.pth", map_location=device)
            except:
                    print(f"No pretrained weights found for {fm_name}, please download them and add them to the corresponding folder")

        elif size == "huge":
            fm = spec_vit.SpecViTHuge()
            try:
                checkpoint = torch.load(f"{path}/pretrained_models/SpectralEarth/spec_ViTh_mae.pth", map_location=device)
            except:
                    print(f"No pretrained weights found for {fm_name}, please download them and add them to the corresponding folder")

        fm.load_state_dict(checkpoint, strict=False)

    elif fm_name == "SpecRnDino":
        fm = spec_resnet.SpecResNet50(num_classes=0)
        try:
            checkpoint = torch.load(f"{path}/pretrained_models/SpectralEarth/spec_rn50_dino.pth", map_location=device)
        except:
            print(f"No pretrained weights found for {fm_name}, please download them and add them to the corresponding folder")
        fm.load_state_dict(checkpoint, strict=False)

    elif fm_name == "SpecRnMoco":
        fm = spec_resnet.SpecResNet50(num_classes=0)
        try:
            checkpoint = torch.load(f"{path}/pretrained_models/SpectralEarth/spec_rn50_moco.pth", map_location=device)
        except:
            print(f"No pretrained weights found for {fm_name}, please download them and add them to the corresponding folder")
        fm.load_state_dict(checkpoint, strict=False)
    
    elif fm_name == "Panopticon":
        if H < 224:
            Y = F.interpolate(Y, size=(224,224))
            new_H = 224
        else:
            new_H = 224
        Y = Y[:,:,:new_H, :new_H]

        if version == "student":
            try:
                checkpoint = torch.load(f"{path}/pretrained_models/Panopticon/panopticon_vitb14.pth", map_location=device)["model"]
            except:
                print(f"No pretrained weights found for {fm_name}, please download them and add them to the corresponding folder")
            checkpoint = {
                key.replace("student.backbone.", ""): value
                for key, value in checkpoint.items()
                if key.startswith("student.backbone.")
            }
        else:
            try:
                checkpoint = torch.load(f"{path}/pretrained_models/Panopticon/panopticon_vitb14.pth", map_location=device)["model"]
            except:
                print(f"No pretrained weights found for {fm_name}, please download them and add them to the corresponding folder")
            checkpoint = {
                key.replace("teacher.backbone.", ""): value
                for key, value in checkpoint.items()
                if key.startswith("teacher.backbone.")
            }

        vit_kwargs = dict(
            img_size = 518, # was trained on 224 but 568 is legacy from last large image size epoch in dinov2 training
            patch_size = 14, init_values = 1.0e-05, ffn_layer = 'mlp', block_chunks = 0, qkv_bias = True, proj_bias = True, 
            ffn_bias = True, num_register_tokens = 0, embed_layer = 'PanopticonPE', 
            pe_args = dict(attn_dim = 2304, chnfus_cfg = dict(layer_norm = False, attn_cfg = dict(num_heads = 16))),
        )
        fm = panopticon_vits.__dict__["vit_base"](**vit_kwargs)
        fm.load_state_dict(checkpoint, strict=True)

    elif fm_name == "HyperSL":
        if H > 128:
            new_H = 128
        else:
            new_H = H
        Y = Y[:,:,:new_H, :new_H]

        try:
            checkpoint = torch.load(f'{path}/pretrained_models/HyperSL/10_base_mask95_checkpoint.pt', map_location=device)
        except:
            print(f"No pretrained weights found for {fm_name}, please download them and add them to the corresponding folder")
        weights = OrderedDict()
        for k, v in checkpoint['model'].items():
            name = k[7:]
            weights[name] = v
        
        fm = SpectralSharedEncoder(
                embedding_dim = 256,
                encoder_depth=8,
                decoder_depth=4,
                num_heads=8,
            )
        fm.load_state_dict(weights)

    elif fm_name == "UniverSat":
        if H < 224:
            Y = F.interpolate(Y, size=(224,224))
            new_H = 224
        else:
            new_H = 224
        Y = Y[:,:,:new_H, :new_H]

        modalities_dict = {
            "earthview":  ["neon"],
            "hyperglobal": ["EO1"],
            "spectralearth": ["enmap"],
        }

        try:
            checkpoint = load_file(f"{path}/pretrained_models/UniverSat/model.safetensors")#, map_location=device)
        except:
            print(f"No pretrained weights found for {fm_name}, please download them and add them to the corresponding folder")

        checkpoint = {
                key.replace("model.", ""): value
                for key, value in checkpoint.items()
                if key.startswith("model.")
            }

        fm = buildUniverSat._build_model(size="base", modalities_dict=modalities_dict)
        fm.load_state_dict(checkpoint, strict=False)

    elif fm_name == "RAMEN":
        if H < 224:
            Y = F.interpolate(Y, size=(224,224))
            new_H = 224
        else:
            new_H = 224
        Y = Y[:,:,:new_H, :new_H]

        modalities = ["hsi"]
        modalities_res = {"hsi" : 30}
        modalities_input_size = {"hsi" : new_H}
        all_res = [30.0]
        wavelengths_dict = {"hsi": wavelengths}
        
        try:
            checkpoint = torch.load(f"{path}/pretrained_models/RAMEN/ramen_base_99.ckpt")["state_dict"]#, map_location=device)
        except:
            print(f"No pretrained weights found for {fm_name}, please download them and add them to the corresponding folder")
        
        fm = RamenViT.RamenViT(modalities, modalities_res, modalities_input_size, all_res=all_res, wavelengths=wavelengths_dict,
                          embed_dim=768)
        fm.load_state_dict(checkpoint, strict=False)
    else:
        raise(f"Fm name {fm_name} is not known, use DOFA, SpecAware, Panopticon, HyperFree, HyperSL, HyperSIGMA, SpecViT, SpecRnDino, SpecRnMoco, UniverSat or RAMEN")
    
    return fm, Y, new_H

def reshape_Y(fm_name, Y, new_H=None, A=None):

    if fm_name == "OFAViT" or fm_name == "DOFA" or fm_name == "SpecAware" or fm_name == "MaskedHSIAutoencoderViT" or fm_name == "Panopticon" or fm_name == "DinoVisionTransformer" or fm_name == "UniverSat" or fm_name == "RAMEN" or fm_name == "RamenViT":
        new_H = 224

        if Y.shape[-1] < new_H:
            Y = F.interpolate(Y, size=(new_H,new_H))
        Y = Y[:,:,:new_H, :new_H]
        
        if A != None:
            if A.shape[-1] < new_H:
                A = F.interpolate(A, size=(new_H,new_H))
            A = A[:,:,:new_H, :new_H]

    elif fm_name == "ImageEncoderViT" or fm_name == "HyperFree" or fm_name == "SpecViTBase" or fm_name == "SpecViT" or fm_name == "SpecViTSmall" or fm_name == "SpecViTLarge" or fm_name == "SpecViTHuge":
        if Y.shape[-1] < new_H:
            Y = F.interpolate(Y, size=(new_H,new_H))
        Y = Y[:,:,:new_H, :new_H]
        
        if A != None:
            if A.shape[-1] < new_H:
                A = F.interpolate(A, size=(new_H,new_H))
            A = A[:,:,:new_H, :new_H]

    elif fm_name == "HyperSL" or fm_name == "SpectralSharedEncoder":
        new_H = 128
        Y = Y[:,:,:new_H, :new_H]
        
        if A != None:
            A = A[:,:,:new_H, :new_H]

    else:
        if A != None:
            return Y, A
        else:
            return Y
        
    if A != None:
        return Y, A
    else:
        return Y

def extract_f(fm, Y, new_H, wavelengths, A=None, patch_size=None, GSD=None):
    fm_name = fm.__class__.__name__
    device = Y.device
    
    if A != None:
        Y, A = reshape_Y(fm_name, utils.standardise(Y), new_H, A)
    else:
        Y = reshape_Y(fm_name, utils.standardise(Y), new_H)

    if fm_name == "OFAViT": # DOFA
        features = fm.forward_features(Y, wavelengths)

    elif fm_name == "ImageEncoderViT": # HyperFree
        if GSD == None:
            GSD = 0.456
            ratio = 1024 / Y.shape[2]
            GSD = GSD / ratio
            GSD = torch.tensor([GSD])
        
        if Y.shape[-1] != 1024:
            Y = reshape_Y("HyperFree", Y)
            
        features = fm(Y, input_wavelength=wavelengths, GSD=GSD, test_mode=True)[-1]
        features = features.reshape(256, features.shape[2]**2).squeeze(0)

    elif fm_name == "SpecViTBase" or fm_name == "SpecViT" or fm_name == "SpecViTSmall" or fm_name == "SpecViTLarge" or fm_name == "SpecViTHuge":
        features = fm(Y)
        features = features[:, 1:, :]
        features = features[0].T

    elif fm_name == "MaskedHSIAutoencoderViT":# SpecAware
        features, _ = fm.forward(Y, wavelength=torch.tensor(wavelengths).to(device), fwhm=torch.tensor([1]).to(device))
    
    elif fm_name == "HyperSIGMA_Unmix":
        features = fm.forward_features(Y)

    elif fm_name == "DinoVisionTransformer": # Panopticon
        x_dict = dict(
            imgs = Y,
            chn_ids = torch.tensor(wavelengths).repeat(1,1)
        )
        features = fm.get_intermediate_layers(x_dict, n=[11], return_class_token=False)[0][0].T
    
    elif fm_name == "UniverSat":
        if GSD is None:
            GSD = 20

        with torch.amp.autocast("cuda"):
            with torch.no_grad():
                if patch_size is None:
                    f, _ = fm.encode({"hsi":Y}, wavelengths={"hsi":wavelengths}, latent_grid=new_H**2, input_res={"hsi":GSD}, subpatches={"hsi":1})
                    features = utils.oneD_to_2d(f[0].T)
                else:
                    f, _ = fm.encode({"hsi":Y}, wavelengths={"hsi":wavelengths}, output_grid=patch_size, input_res={"hsi":GSD})
                    features = f[0].T
                torch.cuda.empty_cache()  

    elif fm_name == "SpectralSharedEncoder": # HyperSL
        if patch_size is None:
            patch_size = new_H

        unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)
        Y_init_patch = unfold(Y).permute(0,2,1)
        Y_init_patch = utils.oneD_to_2d(Y_init_patch.reshape(Y_init_patch.shape[1], Y.shape[1], -1))
        features = torch.zeros(fm.embedding_dim, new_H, new_H)
        n = int(new_H/patch_size)

        with torch.amp.autocast("cuda"):
            with torch.no_grad():
                for i, patch in enumerate(Y_init_patch):
                    patch = patch.permute(1,2,0).unsqueeze(0)
                    f, _, _, _, _, _ = fm.encoder_forward(patch, torch.tensor(wavelengths), 0.)
                    f = utils.oneD_to_2d(f.permute(1,2,0))
                    features[:, patch_size*(i//n):patch_size*(i//n)+patch_size, patch_size*(i%n):patch_size*(i%n)+patch_size] = f
                    del f
                    torch.cuda.empty_cache()
    
    elif fm_name == "RamenViT": # RAMEN
        x = {"hsi":Y.unsqueeze(0)}
        effective_res = 30.0
        features = fm.forward_proj(x, effective_res)["tokens"][0].T

    else:
        raise("Fm is not known, use DOFA, HyperFree, HyperSIGMA, SpecViT, SpecRnDino, SpecRnMoco, Panopticon, UniverSat or RAMEN")
    
    if A != None:
        return Y, A, features
    else:
        return Y, features
