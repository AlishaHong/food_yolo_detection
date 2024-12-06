import os
import random
import cv2
import numpy as np
import json



def create_composite_images_with_polygon(input_dir, output_dir, coordinates, class_folders, num_images):
    # 결과물을 저장하기 위해 output 경로 폴더를 생성
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    canvas_size = (640, 640)  # 캔버스 크기
    used_folders = {key: set() for key in class_folders}  # 클래스별로 사용된 폴더 관리

    for img_idx in range(num_images):
        canvas = np.ones((canvas_size[1], canvas_size[0], 3), dtype=np.uint8) * 255
        shapes = []  # JSON 파일에 저장될 폴리곤 좌표 데이터
        selected_images = {'rice': None, 'soup': None, 'side': []}

        # 각 클래스(rice, soup, side)에 대해 이미지 선택
        for class_label, folders in class_folders.items():
            if class_label == 'side':
                # 반찬은 4개 선택
                side_candidates = []

                for folder in folders:
                    folder_path = os.path.join(input_dir, folder)
                    if not os.path.exists(folder_path):
                        continue

                    images = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                    if not images:
                        continue

                    # 사용되지 않은 이미지를 무작위로 선택
                    selected_image = random.choice(images)
                    side_candidates.append(os.path.join(folder_path, selected_image))

                # 반찬 후보가 4개 미만인 경우 처리
                if len(side_candidates) < 4:
                    raise ValueError("Not enough side dish folders to select 4 unique images.")

                # 무작위로 4개 선택하고 중복 방지
                selected_images['side'] = random.sample(side_candidates, 4)

            else:
                # 밥과 국은 각 폴더에서 이미지를 하나씩 선택
                available_folders = [folder for folder in folders if folder not in used_folders[class_label]]
                if not available_folders:
                    # 모든 폴더가 이미 사용된 경우, 사용된 폴더 초기화
                    used_folders[class_label] = set()
                    available_folders = folders

                # 사용되지 않은 폴더 중 하나를 선택
                selected_folder = random.choice(available_folders)
                used_folders[class_label].add(selected_folder)

                folder_path = os.path.join(input_dir, selected_folder)
                images = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                if not images:
                    continue

                # 폴더에서 무작위로 이미지 선택
                selected_image = random.choice(images)
                selected_images[class_label] = os.path.join(folder_path, selected_image)

        # 이미지 배치 및 JSON 저장 코드 (기존 로직 유지)
        for class_label, image_paths in selected_images.items():
            if class_label in ['rice', 'soup']:
                image_paths = [image_paths]

            for i, image_path in enumerate(image_paths):
                json_path = os.path.splitext(image_path)[0] + ".json"
                if not os.path.exists(json_path):
                    print(f"JSON file not found for image: {image_path}")
                    continue

                original_image = cv2.imread(image_path)
                if original_image is None:
                    print(f"Failed to read image: {image_path}")
                    continue

                original_h, original_w = original_image.shape[:2]
                if class_label == 'rice':
                    x, y, w, h = coordinates['rice']
                elif class_label == 'soup':
                    x, y, w, h = coordinates['soup']
                elif class_label == 'side':
                    x, y, w, h = coordinates['side'][i]

                resized_image = cv2.resize(original_image, (w, h), interpolation=cv2.INTER_AREA)
                canvas[y:y + h, x:x + w] = resized_image

                with open(json_path, 'r') as jf:
                    json_data = json.load(jf)

                for shape in json_data['shapes']:
                    transformed_points = []
                    for point in shape['points']:
                        old_x, old_y = point
                        new_x = x + (old_x / original_w) * w
                        new_y = y + (old_y / original_h) * h
                        transformed_points.append([new_x, new_y])

                    shapes.append({
                        "label": shape['label'],
                        "points": transformed_points,
                        "group_id": shape.get('group_id', None),
                        "shape_type": shape['shape_type'],
                        "flags": shape['flags']
                    })

        output_image_name = f"composite_image_{img_idx + 1}.jpg"
        output_image_path = os.path.join(output_dir, output_image_name)
        cv2.imwrite(output_image_path, canvas)

        output_json_name = os.path.splitext(output_image_name)[0] + ".json"
        output_json_path = os.path.join(output_dir, output_json_name)
        output_json_data = {
            "version": "0.4.15",
            "flags": {},
            "shapes": shapes,
            "imagePath": output_image_name,
            "imageData": None,
            "imageHeight": canvas_size[1],
            "imageWidth": canvas_size[0]
        }

        with open(output_json_path, 'w') as jf:
            json.dump(output_json_data, jf, indent=4)

        print(f"Saved composite image and JSON: {output_image_path}, {output_json_path}")


# 새로운 좌표 설정 (640x640 캔버스에 6개 이미지 배치)
coordinates = {
    'rice': (0, 256, 320, 384),  # 하단 왼쪽 
    'soup': (320, 256, 320, 384), # 하단 오른쪽
    'side': [
        (0, 0, 160, 256),    #상단1구
        (160, 0, 160, 256),  #상단2구
        (320, 0, 160, 256),  #상단3구
        (480, 0, 160, 256)   #상단4구
    ]
}

#이미지 경로와 저장경로
input_directory = "C:/Users/Sesame/Desktop/6mix_scaling_images/1st_only"
output_directory = "C:/Users/Sesame/Desktop/6mix_scaling_images/1st_only/test_output"

# class_folders = {
#     'rice': ['01011001', '01012006', '01012002'],
#     'soup': ['04011005', '04011007', '04017001', '04011011'],
#     'side': ['06012004', '07014001', '11013007', '12011008', '06012008', '08011003', '10012001', '11013002', '12011003', '07013003', '11013010']
# }

class_folders = {
    'rice': ['01011001'],
    'soup': ['04017001'],
    'side': ['06012004', '07014001', '11013007', '12011008']
}

# 실행하기
create_composite_images_with_polygon(
    input_dir=input_directory,
    output_dir=output_directory,
    coordinates=coordinates,
    class_folders=class_folders,
    num_images=8000
)