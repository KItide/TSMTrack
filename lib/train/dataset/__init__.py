# <<<<<<< ours
# from .lasot import Lasot
# from .got10k import Got10k
# from .tracking_net import TrackingNet
# from .imagenetvid import ImagenetVID
# from .coco import MSCOCO
# from .coco_seq import MSCOCOSeq
# from .got10k_lmdb import Got10k_lmdb
# from .lasot_lmdb import Lasot_lmdb
# from .imagenetvid_lmdb import ImagenetVID_lmdb
# from .coco_seq_lmdb import MSCOCOSeq_lmdb
# from .tracking_net_lmdb import TrackingNet_lmdb
# # RGBT dataloader
# from .lasher import LasHeR
# from .vtuav import VTUAV
# # RGBD dataloader
# from .depthtrack import DepthTrack
# # Event dataloader
# from .visevent import VisEvent

def _optional_import(module_name, class_name):
    try:
        module = __import__(f"{__name__}.{module_name}", fromlist=[class_name])
        return getattr(module, class_name)
    except ImportError:
        return None


Lasot = _optional_import("lasot", "Lasot")
Got10k = _optional_import("got10k", "Got10k")
TrackingNet = _optional_import("tracking_net", "TrackingNet")
ImagenetVID = _optional_import("imagenetvid", "ImagenetVID")
MSCOCO = _optional_import("coco", "MSCOCO")
MSCOCOSeq = _optional_import("coco_seq", "MSCOCOSeq")
Got10k_lmdb = _optional_import("got10k_lmdb", "Got10k_lmdb")
Lasot_lmdb = _optional_import("lasot_lmdb", "Lasot_lmdb")
ImagenetVID_lmdb = _optional_import("imagenetvid_lmdb", "ImagenetVID_lmdb")
MSCOCOSeq_lmdb = _optional_import("coco_seq_lmdb", "MSCOCOSeq_lmdb")
TrackingNet_lmdb = _optional_import("tracking_net_lmdb", "TrackingNet_lmdb")
LasHeR = _optional_import("lasher", "LasHeR")
VTUAV = _optional_import("vtuav", "VTUAV")
DepthTrack = _optional_import("depthtrack", "DepthTrack")
VisEvent = _optional_import("visevent", "VisEvent")

