from mmseg.datasets.basesegdataset import BaseSegDataset
from mmseg.registry import DATASETS

@DATASETS.register_module()
class CelebAMaskHQ(BaseSegDataset):
    METAINFO = dict(
        classes = ('Background', 'Skin', 'Nose', 'Eye glasses', 'Left eye', 'Right eye', 'Left brow', 
                'Right brow', 'Left ear', 'Right ear', 'Mouth', 'Upper lip', 
                'Lower lip', 'Hair', 'Hat', 'Ear ring', 'Necklace', 'Neck', 'Cloth'
        ),
        palette = [
                [0, 0, 0], [204, 0, 0], [76, 153, 0], [204, 204, 0], [51, 51, 255], [204, 0, 204], 
                [0, 255, 255], [255, 204, 204], [102, 51, 0], [255, 0, 0], 
                [102, 204, 0], [255, 255, 0], [0, 0, 153], [0, 0, 204], [255, 51, 153], 
                [0, 204, 204], [0, 51, 0], [255, 153, 51], [0, 204, 0]
        ]
    )
    
    def __init__(
        self,
        img_suffix =".jpg",
        seg_map_suffix = ".png",
        reduce_zero_label = False,
        **kwargs
    )->None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix = seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs
        )