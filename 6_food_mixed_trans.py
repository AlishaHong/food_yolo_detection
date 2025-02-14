import os
import random
import cv2
import numpy as np
import json

def create_composite_images_with_polygon(input_dir, output_dir, coordinates, class_folders, num_images):
    """
    여러 클래스의 이미지를 결합하여 하나의 640x640 크기의 캔버스에 배치하고,
    각 이미지의 폴리곤 좌표를 변환하여 JSON 파일로 저장합니다.
    JSON 파일의 label을 원본 그대로 사용합니다.

    Args:
        input_dir (str): 클래스별 이미지가 저장된 폴더 경로.
        output_dir (str): 생성된 이미지를 저장할 폴더 경로.
        coordinates (dict): 각 클래스별 이미지를 배치할 좌표 및 크기 {'rice': (x, y, w, h), ...}.
        class_folders (dict): 클래스별 폴더 이름 {'rice': [...], ...}.
        num_images (int): 생성할 이미지의 총 개수.
    """

    # 결과물을 저장하기 위해 output 경로 폴더를 생성
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # yolo 학습에 적합한 640*640 사이즈로 캔버스 생성
    canvas_size = (640, 640)
    # 음식이미지가 중복되는것을 방지하기 위해 사용된 이미지는 set으로 관리함 
    # 뒤에서 used_images를 활용하여 사용되지 않은 이미지를 필터링함
    used_images = set()

    # 원하는 결과 이미지만큼 반복(메인 반복문)
    for img_idx in range(num_images):
        # 흰색 캔버스 추가
        canvas = np.ones((canvas_size[1], canvas_size[0], 3), dtype=np.uint8) * 255
        # 새로 저장될 json 파일의 폴리곤 좌표 데이터를 담을 리스트 초기화 
        shapes = []

        # 합성에 이용될 선택된 이미지를 담아둘 딕셔너리 초기화
        # rice/soup은 1장씩 반찬은 4가지를 담을 예정이어서 리스트로 생성
        selected_images = {'rice': None, 'soup': None, 'side': []}
        # 반찬 후보를 담아둘 리스트 
        side_candidates = []

        # class_folders -> 밥/국/반찬 카테고리에 어떤 클래스가 속하는지 정리한 딕셔너리
        for class_label, folders in class_folders.items():
            # 폴더별 이미지 확인
            for folder_name in folders:
                # 각 클래스 폴더의 경로를 생성
                folder_path = os.path.join(input_dir, folder_name)
                if not os.path.exists(folder_path):
                    continue
                # 각 클래스 폴더 내 이미지 파일을 가져옴 
                images = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                # 사용되지 않은 이미지만 필터링
                available_images = [os.path.join(folder_path, img) for img in images if os.path.join(folder_path, img) not in used_images]

                # 사용가능한 이미지가 없는 경우
                if not available_images:
                    print(f"No available images in {folder_name}")
                    continue

                # 밥과 국을 선택해보자
                if class_label in ['rice', 'soup'] and selected_images[class_label] is None:
                    # 클래스별 사용 가능한 이미지 후보
                    candidates = []

                    # 밥/국 클래스의 폴더를 순회
                    for folder in class_folders[class_label]:
                        folder_path = os.path.join(input_dir, folder)
                        if not os.path.exists(folder_path):
                            continue  # 폴더가 없으면 건너뛰기

                        # 현재 폴더에서 사용되지 않은 이미지 중 1개만 선택
                        for img in os.listdir(folder_path):
                            if not img.lower().endswith(('.png', '.jpg', '.jpeg')):
                                continue  # 유효한 이미지 파일만 처리
                            
                            img_path = os.path.join(folder_path, img)
                            
                            # 중복되지 않은 이미지만 후보에 추가
                            if img_path not in used_images:
                                candidates.append(img_path)
                                break  # 폴더당 1개의 이미지만 추가

                    # 후보 리스트에서 무작위로 1개 선택
                    if candidates:
                        selected_image = random.choice(candidates)
                        selected_images[class_label] = selected_image
                        used_images.add(selected_image)  # 선택된 이미지는 used_images에 추가
                        # print(f"Selected {class_label}: {selected_image}")
                    else:
                        print(f"No available images for {class_label} across all folders.")

                if class_label == 'side':
                    selected_image = random.choice(available_images)
                    side_candidates.append(selected_image)

        # 위에서 담아둔 반찬 후보지가 4개이상 담겨있다면 
        # print(f'side_candidates: {side_candidates}')  각 클래스에서 하나씩 가져오고 있는 것 확인
        if len(side_candidates) < 4:
            raise ValueError("Not enough side dish folders to select 4 unique images.")
        # 후보 이미지 중 4장을 선택하여 반찬 선택이미지로 담는다.
        selected_images['side'] = random.sample(side_candidates, 4)

        # 선택된 이미지의 레이블과 이미지 경로를 가져와서
        for class_label, image_paths in selected_images.items():
            if class_label in ['rice', 'soup']:
                image_paths = [image_paths]

            # 새로운 json파일의 경로를 만들어 주기 위해 이미지에서 확장자를 제외한 부분을 추출하여
            # .json을 붙여 경로 생성
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
                # 좌표 설정(카테고리에 따라 다른 위치에 배치)
                if class_label == 'rice':
                    x, y, w, h = coordinates['rice']
                elif class_label == 'soup':
                    x, y, w, h = coordinates['soup']
                elif class_label == 'side':
                    x, y, w, h = coordinates['side'][i]

                # 이미지를 지정된 크기로 리사이즈
                resized_image = cv2.resize(original_image, (w, h), interpolation=cv2.INTER_AREA)
                # 캔버스에 이미지 배치하기
                canvas[y:y + h, x:x + w] = resized_image

                # 원본 json파일 읽어오기
                with open(json_path, 'r') as jf:
                    json_data = json.load(jf)
                # json 파일에서 폴리곤 좌표를 가져옴
                for shape in json_data['shapes']:
                    transformed_points = []
                    for point in shape['points']:
                        # 원본 이미지의 좌표를
                        old_x, old_y = point
                        # 캔버스 좌표로 변환
                        # 원본 이미지의 크기 대비 좌표 비율을 계산한다.
                        # 예를들어, old_x가 100이고 원본 이미지의 너비(original_w)가 500이라면 
                        # 좌표비율은 0.2이고 
                        # 이미지 크기가 바뀌어도 상대적으로 비율은 유지되므로 새로운 x를 계산할 수 있다.

                        # 좌표비율 = old_x/original_w(ex-0.2)
                        # 좌표비율 * 새 이미지 너비(ex-0.2*100 = 20)
                        # 시작점 + 위에서 계산한 값을 더해주면 새로운 좌표를 구할 수 있다.
                        new_x = x + (old_x / original_w) * w
                        new_y = y + (old_y / original_h) * h
                        # 변환된 좌표
                        transformed_points.append([new_x, new_y])
                    # json파일의 shapes리스트에 추가할 항목
                    shapes.append({
                        "label": shape['label'],  # 원본 JSON의 label을 그대로 사용
                        "points": transformed_points,   # 변환된 좌표로 변경해줌
                        "group_id": shape.get('group_id', None),
                        "shape_type": shape['shape_type'],
                        "flags": shape['flags']
                    })

        # 결과 이미지 이름
        output_image_name = f"composite_image_{img_idx + 1}.jpg"
        # 결과 이미지 경로
        output_image_path = os.path.join(output_dir, output_image_name)
        # 저장
        cv2.imwrite(output_image_path, canvas)

        # 결과 json파일 이름
        output_json_name = os.path.splitext(output_image_name)[0] + ".json"
        # 결과 json파일 경로
        output_json_path = os.path.join(output_dir, output_json_name)
        # 결과 json파일 내용
        output_json_data = {
            "version": "0.4.15",
            "flags": {},
            "shapes": shapes,
            "imagePath": output_image_name,
            "imageData": None,
            "imageHeight": canvas_size[1],
            "imageWidth": canvas_size[0]
        }
        # 파일을 쓰기 모드로 열어줌
        with open(output_json_path, 'w') as jf:
            # 파이썬 객체를 json 형식으로 변환한 뒤 파일에 기록
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
input_directory = "C:/Users/Sesame/Desktop/6mix_scaling_images/1st_2nd_mix_scaled"
output_directory = "C:/Users/Sesame/Desktop/6mix_scaling_images/1st_2nd_mix_scaled/test_output"

class_folders = {
    'rice': ['01011001', '01012006', '01012002', '03011011'],
    'soup': ['04011005', '04011007', '04017001', '04011011'],
    'side': ['06012004', '07014001', '11013007', '12011008', '06012008', '08011003', 
             '10012001', '11013002', '12011003', '07013003', '11013010', '08012001']
}

# 실행하기
create_composite_images_with_polygon(
    input_dir=input_directory,
    output_dir=output_directory,
    coordinates=coordinates,
    class_folders=class_folders,
    num_images=10
)
