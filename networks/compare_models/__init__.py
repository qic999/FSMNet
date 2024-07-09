from .unet import build_model as UNET
from .mmunet import build_model as MUNET
from .humus_net import build_model as HUMUS
from .MINet import build_model as MINET
from .SANet import build_model as SANET
from .mtrans_net import build_model as MTRANS
from .unimodal_transformer import build_model as TRANS
from .cnn_transformer_new import build_model as CNN_TRANS
from .cnn import build_model as CNN
from .unet_transformer import build_model as UNET_TRANSFORMER
from .swin_transformer import build_model as SWIN_TRANS
from .restormer import build_model as RESTORMER


from .cnn_late_fusion import build_model as MULTI_CNN
from .mmunet_late_fusion import build_model as MMUNET_LATE
from .mmunet_early_fusion import build_model as MMUNET_EARLY
from .trans_unet.trans_unet import build_model as TRANSUNET
from .SwinFusion import build_model as SWINMULTI
from .mUnet_transformer import build_model as MUNET_TRANSFORMER
from .munet_multi_transfuse import build_model as MUNET_TRANSFUSE
from .munet_swinfusion import build_model as MUNET_SWINFUSE
from .munet_multi_concat import build_model as MUNET_CONCAT
from .munet_concat_decomp import build_model as MUNET_CONCAT_DECOMP
from .munet_multi_sum import build_model as MUNET_SUM
# from .mUnet_ARTfusion_new import build_model as MUNET_ART_FUSION
from .mUnet_ARTfusion import build_model as MUNET_ART_FUSION
from .mUnet_Restormer_fusion import build_model as MUNET_RESTORMER_FUSION
from .mUnet_ARTfusion_SeqConcat import build_model as MUNET_ART_FUSION_SEQ

from .ARTUnet import build_model as ART
from .Unet_ART import build_model as UNET_ART
from .unet_restormer import build_model as UNET_RESTORMER
from .mmunet_ART import build_model as MMUNET_ART
from .mmunet_restormer import build_model as MMUNET_RESTORMER
from .mmunet_restormer_v2 import build_model as MMUNET_RESTORMER_V2
from .mmunet_restormer_3blocks import build_model as MMUNET_RESTORMER_SMALL

from .mARTUnet import build_model as ART_MULTI_INPUT

from .ART_Restormer import build_model as ART_RESTORMER
from .ART_Restormer_v2 import build_model as ART_RESTORMER_V2
from .swinIR import build_model as SWINIR

from .Kspace_ConvNet import build_model as KSPACE_CONVNET
from .Kspace_mUnet_new import build_model as KSPACE_MUNET
from .kspace_mUnet_concat import build_model as KSPACE_MUNET_CONCAT
from .kspace_mUnet_AttnFusion import build_model as KSPACE_MUNET_ATTNFUSE

from .DuDo_mUnet import build_model as DUDO_MUNET
from .DuDo_mUnet_ARTfusion import build_model as DUDO_MUNET_ARTFUSION
from .DuDo_mUnet_CatFusion import build_model as DUDO_MUNET_CONCAT


model_factory = {
    'unet_single': UNET,
    'humus_single': HUMUS,
    'transformer_single': TRANS,
    'cnn_transformer': CNN_TRANS,
    'cnn_single': CNN,
    'swin_trans_single': SWIN_TRANS,
    'trans_unet': TRANSUNET,
    'unet_transformer': UNET_TRANSFORMER,
    'restormer': RESTORMER,
    'unet_art': UNET_ART,
    'unet_restormer': UNET_RESTORMER,
    'art': ART,
    'art_restormer': ART_RESTORMER,
    'art_restormer_v2': ART_RESTORMER_V2,
    'swinIR': SWINIR,


    'munet_transformer': MUNET_TRANSFORMER,
    'munet_transfuse': MUNET_TRANSFUSE,
    'cnn_late_multi': MULTI_CNN,
    'unet_multi': MUNET,
    'unet_late_multi':MMUNET_LATE,
    'unet_early_multi':MMUNET_EARLY,
    'munet_ARTfusion': MUNET_ART_FUSION,
    'munet_restormer_fusion': MUNET_RESTORMER_FUSION,


    'minet_multi': MINET,
    'sanet_multi': SANET,
    'mtrans_multi': MTRANS,
    'swin_fusion': SWINMULTI,
    'munet_swinfuse': MUNET_SWINFUSE,
    'munet_concat': MUNET_CONCAT,
    'munet_concat_decomp': MUNET_CONCAT_DECOMP,
    'munet_sum': MUNET_SUM,
    'mmunet_art': MMUNET_ART,
    'mmunet_restormer': MMUNET_RESTORMER,
    'mmunet_restormer_small': MMUNET_RESTORMER_SMALL,
    'mmunet_restormer_v2': MMUNET_RESTORMER_V2,

    'art_multi_input': ART_MULTI_INPUT,

    'munet_ARTfusion_SeqConcat': MUNET_ART_FUSION_SEQ,

    'kspace_ConvNet': KSPACE_CONVNET,
    'kspace_munet': KSPACE_MUNET,
    'kspace_munet_concat': KSPACE_MUNET_CONCAT,
    'kspace_munet_AttnFusion': KSPACE_MUNET_ATTNFUSE,

    'dudo_munet': DUDO_MUNET,
    'dudo_munet_ARTfusion': DUDO_MUNET_ARTFUSION,
    'dudo_munet_concat': DUDO_MUNET_CONCAT,
}


def build_model_from_name(args):
    assert args.model_name in model_factory.keys(), 'unknown model name'

    return model_factory[args.model_name](args)
