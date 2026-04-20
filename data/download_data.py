import os
import urllib.request
import zipfile
import shutil
import json
import stat
import glob
import xml.etree.ElementTree as ET

def sanitize_name(name):
    """Replace Windows-invalid characters"""
    valid = name.replace('?', '_').replace(':', '_').replace('*', '_')
    valid = valid.replace('"', '_').replace('<', '_').replace('>', '_').replace('|', '_')
    return valid

def convert_voc_to_coco(xml_dir, output_json):
    """The original dataset provides Pascal VOC (.xml). We convert it cleanly into COCO (.json)."""
    print(f"Converting VOC XMLs in {xml_dir} to COCO format...")
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    # 1. Gather distinct classes
    classes = set()
    xml_files = glob.glob(os.path.join(xml_dir, "*.xml"))
    for xml_file in xml_files:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for obj in root.findall('object'):
            name = obj.find('name')
            if name is not None and name.text: 
                classes.add(name.text)
            
    # Always sort to ensure deterministic category mapping
    category_map = {name: i for i, name in enumerate(sorted(classes))}
    for name, cid in category_map.items():
        coco_data["categories"].append({"id": cid, "name": name, "supercategory": "none"})
        
    ann_id = 0
    img_id = 0
    
    # 2. Extract bounding boxes per image
    for xml_file in xml_files:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # Calculate matching image filename directly from the extracted name to ensure 100% OS integrity
        base = os.path.splitext(os.path.basename(xml_file))[0]
        img_file = None
        for ext in ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']:
            if os.path.exists(os.path.join(xml_dir, base + ext)):
                img_file = base + ext
                break
                
        if not img_file:
            # Check XML filename node
            file_node = root.find('filename')
            if file_node is not None and file_node.text:
                sanitized_xml_filename = sanitize_name(file_node.text)
                if os.path.exists(os.path.join(xml_dir, sanitized_xml_filename)):
                    img_file = sanitized_xml_filename
            
        if not img_file:
            continue # Silent skip if no paired image exists
            
        size_node = root.find('size')
        if size_node is not None:
            width = int(size_node.find('width').text)
            height = int(size_node.find('height').text)
        else:
            width, height = 0, 0
            
        coco_data["images"].append({
            "id": img_id,
            "file_name": img_file,
            "width": width,
            "height": height
        })
        
        for obj in root.findall('object'):
            name = obj.find('name')
            if name is None or not name.text: continue
            
            bndbox = obj.find('bndbox')
            if bndbox is None: continue
            
            # VOC is xmin, ymin, xmax, ymax. COCO is xmin, ymin, width, height
            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)
            w = xmax - xmin
            h = ymax - ymin
            
            coco_data["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": category_map[name.text],
                "bbox": [xmin, ymin, w, h],
                "area": w * h,
                "iscrowd": 0
            })
            ann_id += 1
            
        img_id += 1
        
    with open(output_json, "w") as f:
        json.dump(coco_data, f)
    print(f" -> Generated {os.path.basename(output_json)} with {len(coco_data['images'])} images and {len(coco_data['annotations'])} annotations.")

def force_delete_readonly(func, path, exc_info):
    if not os.access(path, os.W_OK):
        os.chmod(path, stat.S_IWUSR)
        func(path)

def download_data():
    repo_url = "https://github.com/pratikkayal/PlantDoc-Object-Detection-Dataset/archive/refs/heads/master.zip"
    zip_path = "dataset.zip"
    extract_dir = "PlantDoc-Object-Detection-Dataset-master"
    
    # Clean old artifacts
    if os.path.exists("PlantDoc-Object-Detection-Dataset"):
        shutil.rmtree("PlantDoc-Object-Detection-Dataset", onerror=force_delete_readonly)
        
    print(f"Downloading dataset ZIP from {repo_url}...")
    if not os.path.exists(zip_path):
        urllib.request.urlretrieve(repo_url, zip_path)
    
    print("Extracting files & patching invalid Windows filenames...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for member in zip_ref.infolist():
            parts = member.filename.split('/')
            sanitized_parts = [sanitize_name(p) for p in parts]
            sanitized_name = '/'.join(sanitized_parts)
            target_path = os.path.join(".", sanitized_name)
            
            if member.filename.endswith('/'):
                os.makedirs(target_path, exist_ok=True)
                continue
                
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            with zip_ref.open(member) as source, open(target_path, "wb") as target:
                while True:
                    buf = source.read(1024 * 64) # Read in 64KB chunks
                    if not buf:
                        break
                    target.write(buf)

    print("Structuring directories and mapping VOC to COCO format...")
    os.makedirs(os.path.join("data", "raw"), exist_ok=True)
    
    train_source = os.path.join(extract_dir, "TRAIN")
    test_source = os.path.join(extract_dir, "TEST")
    train_dest = os.path.join("data", "raw", "train")
    test_dest = os.path.join("data", "raw", "test")
    
    if os.path.exists(train_source):
        if os.path.exists(train_dest): shutil.rmtree(train_dest, onerror=force_delete_readonly)
        shutil.move(train_source, train_dest)
        print(f"Moved TRAIN to {train_dest}")
        convert_voc_to_coco(train_dest, os.path.join(train_dest, "_annotations.coco.json"))
        
    if os.path.exists(test_source):
        if os.path.exists(test_dest): shutil.rmtree(test_dest, onerror=force_delete_readonly)
        shutil.move(test_source, test_dest)
        print(f"Moved TEST to {test_dest}")
        convert_voc_to_coco(test_dest, os.path.join(test_dest, "_annotations.coco.json"))
    
    # Cleanup Temp Artifacts
    print("Cleaning up temporary artifact files...")
    if os.path.exists(zip_path): os.remove(zip_path)
    if os.path.exists(extract_dir): shutil.rmtree(extract_dir, onerror=force_delete_readonly)
            
    print("\n✅ Data Sourcing Complete! The cleaned dataset is ready at `data/raw/` !")

if __name__ == "__main__":
    download_data()
