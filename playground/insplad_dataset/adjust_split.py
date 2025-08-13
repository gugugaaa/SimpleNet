import os
import shutil
from PIL import Image
import numpy as np

base_dir = 'F:/ManMadeRam/fake_insplad/src/insplad'
class_name = 'polymer-insulator-lower-shackle'
src_class_dir = os.path.join(base_dir, class_name)
dst_base_dir = 'F:/ManMadeRam/fake_insplad/adjust_split/insplad'
dst_class_dir = os.path.join(dst_base_dir, class_name)

def get_img_hw_avg(img_path):
    with Image.open(img_path) as img:
        w, h = img.size
    return (w + h) / 2

def collect_images(folder):
    img_list = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
                img_path = os.path.join(root, f)
                img_list.append(img_path)
    return img_list

def select_nearest_imgs(img_paths, avg_value, num=50):
    img_hw = [(p, abs(get_img_hw_avg(p) - avg_value)) for p in img_paths]
    img_hw.sort(key=lambda x: x[1])
    selected = [x[0] for x in img_hw[:num]]
    rest = [x[0] for x in img_hw[num:]]
    return selected, rest

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def main():
    # 1. 收集test/good和test/defect*图片
    test_good_dir = os.path.join(src_class_dir, 'test', 'good')
    test_good_imgs = collect_images(test_good_dir)
    test_defect_dirs = []
    for d in os.listdir(os.path.join(src_class_dir, 'test')):
        if d != 'good':
            test_defect_dirs.append(os.path.join(src_class_dir, 'test', d))
    test_defect_imgs = []
    defect_map = {}
    for defect_dir in test_defect_dirs:
        imgs = collect_images(defect_dir)
        test_defect_imgs.extend(imgs)
        defect_map[defect_dir] = imgs

    # 2. 计算所有图片的宽高平均值
    all_imgs = test_good_imgs + test_defect_imgs
    all_hw = [get_img_hw_avg(p) for p in all_imgs]
    overall_avg = np.mean(all_hw)

    # 3. 选取每个类别最接近平均值的50张
    selected_good, rest_good = select_nearest_imgs(test_good_imgs, overall_avg, 50)
    selected_defect = {}
    rest_defect = {}
    for defect_dir, imgs in defect_map.items():
        sel, rest = select_nearest_imgs(imgs, overall_avg, 50)
        selected_defect[defect_dir] = sel
        rest_defect[defect_dir] = rest

    # 4. 复制到新目录
    # 4.1 复制train/good
    src_train_good = os.path.join(src_class_dir, 'train', 'good')
    dst_train_good = os.path.join(dst_class_dir, 'train', 'good')
    ensure_dir(dst_train_good)
    for img in collect_images(src_train_good):
        shutil.copy2(img, dst_train_good)
    # 4.2 将test/good剩余图片移动到train/good
    for img in rest_good:
        shutil.copy2(img, dst_train_good)
    # 4.3 复制test/good选中图片
    dst_test_good = os.path.join(dst_class_dir, 'test', 'good')
    ensure_dir(dst_test_good)
    for img in selected_good:
        shutil.copy2(img, dst_test_good)
    # 4.4 复制test/defect*选中图片
    for defect_dir, imgs in selected_defect.items():
        defect_name = os.path.basename(defect_dir)
        dst_defect_dir = os.path.join(dst_class_dir, 'test', defect_name)
        ensure_dir(dst_defect_dir)
        for img in imgs:
            shutil.copy2(img, dst_defect_dir)
    # 4.5 删除test/defect*剩余图片（不复制到新目录）
    # 不需要操作，直接不复制即可

    print('调整完成，数据已保存到:', dst_class_dir)

if __name__ == '__main__':
    main()
