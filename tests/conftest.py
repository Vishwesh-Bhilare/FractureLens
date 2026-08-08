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
