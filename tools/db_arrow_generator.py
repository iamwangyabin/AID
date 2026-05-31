import sys
import datetime
import shutil
import gc
import os
import json
from multiprocessing import Pool
from tqdm import tqdm
import hashlib
from PIL import Image
import pandas as pd
from datasets import Dataset, concatenate_datasets, load_from_disk
import io


def parse_data(args):
    img_path, dataset_root = args
    img_full_path = os.path.join(dataset_root, img_path)

    try:
        with open(img_full_path, "rb") as fp:
            image = fp.read()
            md5 = hashlib.md5(image).hexdigest()

        with Image.open(img_full_path) as f:
            width, height = f.size
        return {
            "image_path": img_path,
            "md5": md5,
            "width": width,
            "height": height,
            "image": image
        }

        # with Image.open(img_full_path) as f:
        #     width, height = f.size
        #     webp_image = io.BytesIO()
        #     f.save(webp_image, format='WebP', quality=90)
        #     webp_image.seek(0)
        #     md5 = hashlib.md5(webp_image.read()).hexdigest()
        #
        # return {
        #     "image_path": img_path,
        #     "md5": md5,
        #     "width": width,
        #     "height": height,
        #     "image": webp_image.getvalue()
        # }

    except Exception as e:
        print(f'error: {e}')
        return None


def build_and_save_mapping(dataset, output_path):
    mapping = {}

    for index, example in enumerate(dataset):
        image_path = example['image_path']
        mapping[image_path] = index

    with open(output_path, 'w') as file:
        json.dump(mapping, file)

    return mapping


def load_mapping(mapping_path):
    with open(mapping_path, 'r') as file:
        mapping = json.load(file)
    return mapping


def copy_json_files(json_dir, arrow_dir):
    if os.path.isfile(json_dir):
        output_path = os.path.join(arrow_dir, os.path.basename(json_dir))
        if os.path.abspath(json_dir) != os.path.abspath(output_path):
            shutil.copy2(json_dir, output_path)
        return

    for file_name in os.listdir(json_dir):
        if not file_name.endswith('.json'):
            continue
        source_path = os.path.join(json_dir, file_name)
        output_path = os.path.join(arrow_dir, file_name)
        if os.path.abspath(source_path) != os.path.abspath(output_path):
            shutil.copy2(source_path, output_path)


def make_arrow(json_dir, dataset_root, arrow_dir, pool_num):
    print(json_dir)
    print(arrow_dir)
    if not os.path.exists(arrow_dir):
        os.makedirs(arrow_dir)

    image_paths = set()
    if os.path.isfile(json_dir):
        with open(json_dir, 'r') as f:
            data = json.load(f)
        for subset in data.keys():
            for img_path, label in data[subset].items():
                image_paths.add(img_path)
    else:
        for file_name in os.listdir(json_dir):
            if file_name.endswith('.json'):
                file_path = os.path.join(json_dir, file_name)
                with open(file_path, 'r') as f:
                    data = json.load(f)
                for subset in data.keys():
                    for img_path, label in data[subset].items():
                        image_paths.add(img_path)

    image_paths = list(image_paths)
    end_id = len(image_paths)
    start_id = 0
    print(f'start_id:{start_id}  end_id:{end_id}')
    image_paths = image_paths[start_id:end_id]
    num_slice = 10000
    start_sub = int(start_id / num_slice)
    sub_len = int(len(image_paths) // num_slice)
    subs = list(range(sub_len + 1))

    temp_dirs = []

    for sub in tqdm(subs):
        temp_arrow_dir = os.path.join(arrow_dir, f'tmp_{sub + start_sub}')
        if not os.path.exists(temp_arrow_dir):
            os.makedirs(temp_arrow_dir)
        sub_data = image_paths[sub * num_slice: (sub + 1) * num_slice]
        sub_data_with_params = [(img_path, dataset_root) for img_path in sub_data]

        with Pool(pool_num) as pool:
            bs = pool.map(parse_data, sub_data_with_params)
            bs = [b for b in bs if b]

        print(f'length of this batch: {len(bs)}')

        dataframe = pd.DataFrame(bs)
        intermediate_dataset = Dataset.from_pandas(dataframe)
        intermediate_dataset.save_to_disk(temp_arrow_dir)
        temp_dirs.append(temp_arrow_dir)

        del dataframe
        del intermediate_dataset
        del bs
        gc.collect()

    # 合并所有临时文件
    all_datasets = [Dataset.load_from_disk(temp_dir) for temp_dir in temp_dirs]
    final_dataset = concatenate_datasets(all_datasets)
    final_dataset.save_to_disk(arrow_dir)

    # 清理临时目录
    for temp_dir in temp_dirs:
        shutil.rmtree(temp_dir)


    # 构建并保存映射
    dataset = load_from_disk(arrow_dir)
    output_path = os.path.join(arrow_dir, 'mapping.json')
    image_path_to_index = build_and_save_mapping(dataset, output_path)
    copy_json_files(json_dir, arrow_dir)


if __name__ == '__main__':
    if len(sys.argv) != 5:
        print("Usage: python tools/db_arrow_generator.py ${json_dir} ${dataset_root} ${output_arrow_dir} ${pool_num}")
        print("json_dir: The directory containing your JSON files.")
        print("dataset_root: The root directory where images are stored.")
        print("output_arrow_dir: The path for storing the created Arrow file.")
        print("pool_num: The number of processes, used for multiprocessing. If you encounter memory issues, you can set pool_num to 1.")
        sys.exit(1)

    json_dir = sys.argv[1]
    dataset_root = sys.argv[2]
    output_arrow_dir = sys.argv[3]
    pool_num = int(sys.argv[4])

    make_arrow(json_dir, dataset_root, output_arrow_dir, pool_num)
