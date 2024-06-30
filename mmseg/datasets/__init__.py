# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications: Add additional datasets


from .builder import DATASETS, PIPELINES, build_dataloader, build_dataset
from .custom import CustomDataset
from .dataset_wrappers import ConcatDataset, RepeatDataset

from .lesion_dataset import LesionDataset

__all__ = [
    'CustomDataset',
    'build_dataloader',
    'ConcatDataset',
    'RepeatDataset',
    'DATASETS',
    'build_dataset',
    'PIPELINES',
    'CityscapesDataset',
    'GTADataset',
    'SynthiaDataset',
    'UDADataset',
    'ACDCDataset',
    'DarkZurichDataset',
    'GTASplitDataset',
    'SYNTHIASplitDataset',
    'CityscapesSplitDataset',

    'LesionDataset',
]
