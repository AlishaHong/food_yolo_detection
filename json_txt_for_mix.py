import json
import os
import shutil
from pathlib import Path


class SAM_Json2Yolo:
    def __init__(self, data_dir, folder_to_label_map):
        self.data_dir = data_dir
        self.folder_to_label_map = folder_to_label_map  # 폴더 이름과 숫자 매핑
        self.cls_names = list(folder_to_label_map.keys())  # 매핑된 폴더 이름을 클래스 이름으로 사용

    def convert(self):
        """JSON 파일 로드 > 변환 전처리 실행."""
        self.move_files()

        json_dir = os.path.join(self.data_dir, 'json')
        json_files = [os.path.join(json_dir, f) for f in os.listdir(json_dir) if f.endswith('.json')]

        for json_path in json_files:
            with open(json_path, encoding='utf-8') as f:
                data = json.load(f)

            if not self.cls_names:
                self.cls_names = self.extract_cls_names(data)
            self.save_cls_names()

            self.preprocess_img(data)

    def move_files(self):
        """원본 디렉토리(data_dir)에 json, img 폴더 생성 후 파일 이동."""
        json_dir = os.path.join(self.data_dir, 'json')
        img_dir = os.path.join(self.data_dir, 'images')

        os.makedirs(json_dir, exist_ok=True)
        os.makedirs(img_dir, exist_ok=True)

        for filename in os.listdir(self.data_dir):
            src_path = os.path.join(self.data_dir, filename)
            if filename.endswith('.json'):
                shutil.move(src_path, json_dir)
            elif filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                shutil.move(src_path, img_dir)

    def save_cls_names(self):
        """클래스 목록 > .names로 저장."""
        os.makedirs(self.data_dir, exist_ok=True)
        cls_file_path = os.path.join(self.data_dir, 'classes.names')
        with open(cls_file_path, 'w') as f:
            for cls_name in self.cls_names:
                f.write(f"{cls_name}\n")

    def preprocess_img(self, data):
        """IMG 경로, 폴리곤 좌표, 클래스 ID 매칭 > 라벨링 파일(.txt) 생성."""
        img_name = data['imagePath']
        width = data['imageWidth']
        height = data['imageHeight']

        label_dir = os.path.join(self.data_dir, 'labels')
        os.makedirs(label_dir, exist_ok=True)
        label_file_path = os.path.join(label_dir, f"{Path(img_name).stem}.txt")

        with open(label_file_path, 'w') as label_file:
            for shape in data['shapes']:
                # 클래스 아이디를 폴더 이름 기반으로 설정
                folder_name = shape['label']
                if folder_name in self.folder_to_label_map:
                    cls_id = self.folder_to_label_map[folder_name]
                else:
                    print(f"Unknown label: {folder_name} - skipping.")
                    continue

                polygon_points = shape['points']
                polygon_str = " ".join([f"{x / width:.6f} {y / height:.6f}" for x, y in polygon_points])
                label_file.write(f"{cls_id} {polygon_str}\n")


# 폴더 이름과 숫자를 매핑
folder_to_label_map = {
    '01011001': 0, '04017001': 1, '06012004': 2, '07014001': 3, '11013007': 4, '12011008': 5, 
    '01012006': 6, '04011005': 7, '04011007': 8, '06012008': 9, '08011003': 10, '10012001': 11, '11013002': 12, '12011003': 13, 
    '01012002' : 14, '04011011': 15, '07013003': 16, '11013010': 17
}

# 데이터 디렉토리
# data_dir = "C:/Users/Sesame/food_yolo_detection/6mixed_output"
data_dir = 'C:/Users/Sesame/food_yolo_detection/valid_scaling_output/12011008'

# SAM2 > YOLO 변환
converter = SAM_Json2Yolo(data_dir, folder_to_label_map)
converter.convert()
