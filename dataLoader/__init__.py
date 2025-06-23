from .llff import LLFFDataset
from .blender import BlenderDataset
from .nsvf import NSVF
from .tankstemple import TanksTempleDataset
from .your_own_data import YourOwnDataset
from .llff_json import LLFFJsonDataset



dataset_dict = {'blender': BlenderDataset,
               'llff':LLFFDataset,
               'llff_json':LLFFJsonDataset,
               'tankstemple':TanksTempleDataset,
               'nsvf':NSVF,
                'own_data':YourOwnDataset}