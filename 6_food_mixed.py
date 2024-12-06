import os
import random
import cv2
import numpy as np

def create_composite_images(input_dir, output_dir, coordinates, class_folders, num_images=10):
    """
    한 장의 흰색 캔버스에 국, 밥을 반드시 포함하고, 반찬 리스트에서 랜덤하게 4개를 선택하여 배치.
    이미지 크기를 캔버스 영역에 맞게 조정하여 빈 공간 없이 꽉 차도록 배치.
    반찬은 각 폴더에서 1개씩 선택하여 중복되지 않게 함.
    총 num_images 장의 이미지를 생성.

    Args:
        input_dir (str): 클래스별 이미지가 저장된 폴더 경로.
        output_dir (str): 생성된 이미지를 저장할 폴더 경로.
        coordinates (dict): 클래스별 이미지를 배치할 좌표와 크기 {'rice': (x, y, w, h), 'soup': (x, y, w, h), 'side': [(x, y, w, h), ...]}.
        class_folders (dict): 클래스별 폴더 이름 {'rice': [...], 'soup': [...], 'side': [...]}.
        num_images (int): 생성할 이미지의 총 개수.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 흰색 캔버스 크기
    canvas_size = (1280, 720)

    for img_idx in range(num_images):  # 지정된 수만큼 이미지를 생성
        canvas = np.ones((canvas_size[1], canvas_size[0], 3), dtype=np.uint8) * 255

        # 각 클래스별로 이미지 선택
        selected_images = {
            'rice': None,
            'soup': None,
            'side': []
        }

        # 밥, 국, 반찬 이미지 선택
        side_candidates = []  # 반찬 폴더별로 한 이미지씩 선택
        for class_label, folders in class_folders.items():
            for folder_name in folders:
                folder_path = os.path.join(input_dir, folder_name)
                if not os.path.exists(folder_path):
                    continue

                images = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                if not images:
                    continue

                # 밥과 국은 1개씩 반드시 선택
                if class_label in ['rice', 'soup'] and selected_images[class_label] is None:
                    selected_images[class_label] = os.path.join(folder_path, random.choice(images))

                # 반찬은 폴더별로 하나씩 선택
                if class_label == 'side':
                    side_candidates.append(os.path.join(folder_path, random.choice(images)))

        # 반찬에서 중복 없이 폴더별 1개씩 총 4개 선택
        if len(side_candidates) < 4:
            raise ValueError("Not enough side dish folders to select 4 unique images.")
        selected_images['side'] = random.sample(side_candidates, 4)

        # 캔버스에 이미지 배치
        for class_label, image_paths in selected_images.items():
            if class_label in ['rice', 'soup']:
                image_paths = [image_paths]  # 리스트로 변환
            for i, image_path in enumerate(image_paths):
                original_image = cv2.imread(image_path)
                if original_image is None:
                    print(f"Failed to read image: {image_path}")
                    continue

                # 좌표와 크기 설정
                if class_label == 'rice':
                    x, y, w, h = coordinates['rice']
                elif class_label == 'soup':
                    x, y, w, h = coordinates['soup']
                elif class_label == 'side':
                    x, y, w, h = coordinates['side'][i]  # 반찬 좌표는 순서대로 사용

                # 이미지 크기 조정
                resized_image = cv2.resize(original_image, (w, h), interpolation=cv2.INTER_AREA)

                # 이미지를 캔버스 위에 배치
                canvas[y:y + h, x:x + w] = resized_image

        # 각 이미지 저장
        output_path = os.path.join(output_dir, f"6_mixed_image_{img_idx + 1}.jpg")
        cv2.imwrite(output_path, canvas)
        print(f"Composite image saved: {output_path}")


# 사용 예시
input_directory = "C:/Users/Sesame/food_yolo_detection/6_food_mixed"  # 클래스별 이미지 폴더 경로
output_directory = "C:/Users/Sesame/food_yolo_detection/6_food_mixed/output"  # 생성된 이미지 저장 경로

# 좌표 및 크기 설정
coordinates = {
    'rice': (0, 360, 640, 360),  # 하단 왼쪽 (x, y, width, height)
    'soup': (640, 360, 640, 360),  # 하단 오른쪽 (x, y, width, height)
    'side': [
        (0, 0, 320, 360),    # 상단 1구 (x, y, width, height)
        (320, 0, 320, 360),  # 상단 2구 (x, y, width, height)
        (640, 0, 320, 360),  # 상단 3구 (x, y, width, height)
        (960, 0, 320, 360)   # 상단 4구 (x, y, width, height)
    ]
}

# 클래스별 폴더 이름
class_folders = {
    'rice': ['01011001'],  # 밥이 포함된 폴더 이름
    'soup': ['04017001'],  # 국이 포함된 폴더 이름
    'side': ['06012004', '07014001', '11013007', '12011008']  # 반찬이 포함된 폴더 이름
}

# 함수 실행
create_composite_images(
    input_dir=input_directory,
    output_dir=output_directory,
    coordinates=coordinates,
    class_folders=class_folders,
    num_images=1000  # 생성할 이미지 개수
)
