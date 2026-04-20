# [Rule 95: Transfer learning and adaptation]
# We use the Roboflow API to source the COCO-formatted version of PlantDoc 
# to make it compatible with the DETR architecture.
from roboflow import Roboflow

def download_coco_format():
    rf = Roboflow(api_key="YOUR_FREE_API_KEY")
    project = rf.workspace("joseph-nelson").project("plantdoc")
    dataset = project.version(1).download("coco")
    return dataset

if __name__ == "__main__":
    download_coco_format()