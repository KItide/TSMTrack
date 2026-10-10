class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/mnt/disk2/dwh_documents/TSMTrack'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/mnt/disk2/dwh_documents/TSMTrack/tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = '/mnt/disk2/dwh_documents/TSMTrack/pretrained_networks'
        self.got10k_val_dir = '/home/dfc/Documents/dwh_documents/HMFT/HMFT-master/dataset/got10k/val'
        self.lasot_lmdb_dir = '/home/dfc/Documents/dwh_documents/HMFT/HMFT-master/dataset/lasot_lmdb'
        self.got10k_lmdb_dir = '/home/dfc/Documents/dwh_documents/HMFT/HMFT-master/dataset/got10k_lmdb'
        self.trackingnet_lmdb_dir = '/home/dfc/Documents/dwh_documents/HMFT/HMFT-master/dataset/trackingnet_lmdb'
        self.coco_lmdb_dir = '/home/dfc/Documents/dwh_documents/HMFT/HMFT-master/dataset/coco_lmdb'
        self.coco_dir = '/home/dfc/Documents/dwh_documents/HMFT/HMFT-master/dataset/coco'
        self.lasot_dir = '/home/dfc/Documents/dwh_documents/HMFT/HMFT-master/dataset/lasot'
        self.got10k_dir = '/home/dfc/Documents/dwh_documents/HMFT/HMFT-master/dataset/got10k/train'
        self.trackingnet_dir = '/home/dfc/Documents/dwh_documents/HMFT/HMFT-master/dataset/trackingnet'
        self.depthtrack_dir = '/home/dfc/Documents/dwh_documents/HMFT/HMFT-master/dataset/depthtrack/train'
        self.lasher_dir = '/home/dfc/Documents/dwh_documents/HMFT/HMFT-master/dataset/lasher/TrainingSet/trainingset'
        self.vtuav_dir = '/home/dfc/Documents/dwh_documents/HMFT/HMFT-master/dataset/train_ST'
        self.visevent_dir = '/home/dfc/Documents/dwh_documents/HMFT/HMFT-master/dataset/visevent/train'
