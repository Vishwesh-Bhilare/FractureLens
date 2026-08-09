import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
import SimpleITK as sitk

@pytest.fixture()
def synthetic_root(tmp_path: Path) -> Path:
    root = tmp_path / "data"; (root/"train").mkdir(parents=True); (root/"labels").mkdir()
    label = np.zeros((40,40,40), dtype=np.uint8)
    label[5:15,5:15,5:15] = 21
    label[5:15,5:15,15:25] = 22
    label[20:30,20:30,5:15] = 1
    image = np.zeros_like(label, dtype=np.int16); image[label > 0] = 500
    for arr, folder in ((image,"train"),(label,"labels")):
        img = sitk.GetImageFromArray(arr); img.SetSpacing((1.0,1.0,1.0)); sitk.WriteImage(img, str(root/folder/"001.mha"))
    return root

@pytest.fixture()
def synthetic_symmetric_root(tmp_path: Path) -> Path:
    root = tmp_path / "data"; (root/"train").mkdir(parents=True); (root/"labels").mkdir()
    label = np.zeros((64,64,64), dtype=np.uint8)
    # Sacrum centered at x ~= 31.5 mm.
    label[25:39, 27:37, 29:35] = 1
    # Intact left hip block.
    label[18:38, 16:36, 8:20] = 11
    # Right hip rough mirror split into two fragments.
    label[18:28, 16:36, 44:56] = 21
    label[28:38, 16:36, 44:56] = 22
    image = np.zeros_like(label, dtype=np.int16); image[label > 0] = 500
    for arr, folder in ((image,"train"),(label,"labels")):
        img = sitk.GetImageFromArray(arr); img.SetSpacing((1.0,1.0,1.0)); sitk.WriteImage(img, str(root/folder/"002.mha"))
    return root
