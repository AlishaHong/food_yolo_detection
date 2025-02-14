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
        self.move_files()   # 파일 이동

        # JSON 파일 경로 설정
        json_dir = os.path.join(self.data_dir, 'json')
        json_files = [os.path.join(json_dir, f) for f in os.listdir(json_dir) if f.endswith('.json')]

        # 각 JSON 파일 변환 처리
        for json_path in json_files:
            with open(json_path, encoding='utf-8') as f:
                data = json.load(f) # JSON 파일 로드

            # 클래스 이름이 비어있다면 JSON 데이터에서 추출
            if not self.cls_names:
                self.cls_names = self.extract_cls_names(data)
            self.save_cls_names()   # 클래스 이름 저장

            # 이미지 및 라벨 데이터 처리
            self.preprocess_img(data)

    def move_files(self):
        """원본 디렉토리(data_dir)에 json, img 폴더 생성 후 파일 이동."""
        json_dir = os.path.join(self.data_dir, 'json')
        img_dir = os.path.join(self.data_dir, 'images')

        # json/ 및 images/ 폴더 생성
        os.makedirs(json_dir, exist_ok=True)
        os.makedirs(img_dir, exist_ok=True)

        # 데이터 디렉토리 내 파일 이동
        for filename in os.listdir(self.data_dir):
            src_path = os.path.join(self.data_dir, filename)
            if filename.endswith('.json'):
                shutil.move(src_path, json_dir) # JSON 파일 이동
            elif filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                shutil.move(src_path, img_dir)  # 이미지 파일 이동

    def save_cls_names(self):
        """클래스 목록 > .names로 저장."""
        os.makedirs(self.data_dir, exist_ok=True)
        cls_file_path = os.path.join(self.data_dir, 'classes.names')
        with open(cls_file_path, 'w') as f:
            for cls_name in self.cls_names:
                f.write(f"{cls_name}\n")    # 각 클래스 이름을 한 줄씩 기록

    def preprocess_img(self, data):
        """IMG 경로, 폴리곤 좌표, 클래스 ID 매칭 > 라벨링 파일(.txt) 생성."""
        img_name = data['imagePath']    # 이미지 파일 이름
        width = data['imageWidth']  # 이미지 너비
        height = data['imageHeight']    # 이미지 높이

        # 라벨 파일 저장 경로 설정
        label_dir = os.path.join(self.data_dir, 'labels')
        os.makedirs(label_dir, exist_ok=True)
        label_file_path = os.path.join(label_dir, f"{Path(img_name).stem}.txt")
        
        # 라벨 파일 생성
        with open(label_file_path, 'w') as label_file:
            for shape in data['shapes']:
                # 클래스 아이디를 폴더 이름 기반으로 설정
                folder_name = shape['label']
                if folder_name in self.folder_to_label_map:
                    cls_id = self.folder_to_label_map[folder_name]  # 매핑된 클래스 ID 가져오기
                else:
                    print(f"Unknown label: {folder_name} - skipping.")
                    continue
                
                # 폴리곤 좌표를 YOLO 형식으로 변환
                polygon_points = shape['points']
                polygon_str = " ".join([f"{x / width:.6f} {y / height:.6f}" for x, y in polygon_points])
                label_file.write(f"{cls_id} {polygon_str}\n")


# 폴더 이름과 숫자를 매핑
folder_to_label_map = {
    '01011001': 0, '04017001': 1, '06012004': 2, '07014001': 3, '11013007': 4, '12011008': 5, 
    '01012006': 6, '04011005': 7, '04011007': 8, '06012008': 9, '08011003': 10, '10012001': 11, '11013002': 12, '12011003': 13, 
    '01012002' : 14, '04011011': 15, '07013003': 16, '11013010': 17, '03011011': 18, '08012001': 19
}

# 데이터 디렉토리
# data_dir = "C:/Users/Sesame/food_yolo_detection/6mixed_output"
data_dir = 'C:/Users/Sesame/Desktop/total_100org_50scaled/mix'

# SAM2 > YOLO 변환
converter = SAM_Json2Yolo(data_dir, folder_to_label_map)
converter.convert()
