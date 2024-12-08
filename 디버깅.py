import os
import random
import cv2
import numpy as np
import json

def scale_and_save_images_to_separate_folders(input_dir, output_dir_scale50, output_dir_scale100, coordinates=None, class_folders=None, seed=None):
    """
    이미지를 스케일링 및 원본 저장을 수행하며, 두 가지 작업 결과를 별도의 폴더에 저장.
    - 작업 1: 스케일 50% + 원본 100% => output_dir_scale50
    - 작업 2: 스케일 100% + 원본 50% => output_dir_scale100
    """

    if seed is not None:
        random.seed(seed)  # 랜덤 시드 설정

    if not os.path.exists(output_dir_scale50):
        os.makedirs(output_dir_scale50)

    if not os.path.exists(output_dir_scale100):
        os.makedirs(output_dir_scale100)

    if coordinates is None or class_folders is None:
        raise ValueError("Coordinates and class_folders must be provided.")

    # 캔버스 크기 (640x640)
    canvas_size = (640, 640)

    for class_folder in os.listdir(input_dir):
        class_path = os.path.join(input_dir, class_folder)

        if os.path.isdir(class_path):
            images = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            total_images = len(images)
            print(f"Processing folder: {class_folder}, Total images found: {total_images}")

            if class_folder in class_folders['rice']:
                class_label = 'rice'
            elif class_folder in class_folders['soup']:
                class_label = 'soup'
            elif class_folder in class_folders['side']:
                class_label = 'side'
            else:
                print(f"Skipping unknown class folder: {class_folder}")
                continue

            # 작업 1: 스케일 50% + 원본 100% 저장
            output_folder_scale50 = os.path.join(output_dir_scale50, class_folder)
            os.makedirs(output_folder_scale50, exist_ok=True)

            # 작업 2: 스케일 100% + 원본 50% 저장
            output_folder_scale100 = os.path.join(output_dir_scale100, class_folder)
            os.makedirs(output_folder_scale100, exist_ok=True)

            # 동일한 50% 샘플 선택
            sampled_images = random.sample(images, len(images) // 2)

            # 작업 1: 스케일 50% 저장
            for image_file in sampled_images:
                print(f"[DEBUG] Processing scale 50% for {image_file}")
                process_image(image_file, class_path, output_folder_scale50, coordinates, class_label, canvas_size, scale_type=0.5)

            # 작업 1: 원본 100% 저장
            for image_file in images:
                print(f"[DEBUG] Saving original 100% for {image_file}")
                save_original_image(image_file, class_path, output_folder_scale50)

            # 작업 2: 스케일 100% 저장
            for image_file in images:
                print(f"[DEBUG] Processing scale 100% for {image_file}")
                process_image(image_file, class_path, output_folder_scale100, coordinates, class_label, canvas_size, scale_type=1.0)

            # 작업 2: 원본 50% 저장
            for image_file in sampled_images:
                print(f"[DEBUG] Saving original 50% for {image_file}")
                save_original_image(image_file, class_path, output_folder_scale100)


def process_image(image_file, class_path, output_folder, coordinates, class_label, canvas_size, scale_type):
    """
    이미지 스케일링 및 JSON 저장 처리.
    """
    image_path = os.path.normpath(os.path.join(class_path, image_file))
    json_path = os.path.normpath(os.path.join(class_path, os.path.splitext(image_file)[0] + '.json'))

    print("image_file:", image_file)  # 디버깅 코드
    print("Original JSON path:", json_path)  # 디버깅 코드

    if not os.path.exists(json_path):
        print(f"[WARNING] No JSON file for image: {image_file}. Skipping...")
        return

    if not os.path.exists(image_path):
        print(f"[ERROR] Image file not found: {image_path}. Skipping...")
        return

    try:
        original_image = cv2.imread(image_path)
        if original_image is None:
            print(f"[ERROR] Failed to read image: {image_path}. Skipping...")
            return

        if class_label == 'rice':
            x, y, w, h = coordinates['rice']
        elif class_label == 'soup':
            x, y, w, h = coordinates['soup']
        elif class_label == 'side':
            x, y, w, h = random.choice(coordinates['side'])

        # 캔버스 생성
        canvas = np.ones((canvas_size[1], canvas_size[0], 3), dtype=np.uint8) * 255
        resized_image = cv2.resize(original_image, (w, h), interpolation=cv2.INTER_AREA)
        canvas[y:y + h, x:x + w] = resized_image

        scaled_image_name = os.path.splitext(image_file)[0] + "_scaled" + os.path.splitext(image_file)[1]
        scaled_image_path = os.path.normpath(os.path.join(output_folder, scaled_image_name))
        cv2.imwrite(scaled_image_path, canvas)
        print(f"[INFO] Saved scaled image: {scaled_image_path}")

        # JSON 변환 및 저장
        with open(json_path, 'r') as jf:
            json_data = json.load(jf)

        transformed_points = []
        if 'shapes' in json_data and len(json_data['shapes']) > 0:
            for point in json_data['shapes'][0]['points']:
                old_x, old_y = point
                transformed_x = x + (old_x / original_image.shape[1]) * w
                transformed_y = y + (old_y / original_image.shape[0]) * h
                transformed_points.append([transformed_x, transformed_y])

        scaled_json_data = {
            "version": json_data.get("version", "0.4.15"),
            "flags": {},
            "shapes": [
                {
                    "label": json_data['shapes'][0]['label'],
                    "text": "",
                    "points": transformed_points,
                    "group_id": None,
                    "shape_type": "polygon",
                    "flags": {}
                }
            ],
            "imagePath": scaled_image_name,
            "imageData": None,
            "imageHeight": canvas_size[1],
            "imageWidth": canvas_size[0]
        }

        # JSON 파일 경로를 `output_folder` 기준으로 생성
        scaled_json_path = os.path.normpath(os.path.join(output_folder, os.path.splitext(scaled_image_name)[0] + ".json"))
        with open(scaled_json_path, 'w') as njf:
            json.dump(scaled_json_data, njf, indent=4)
        print(f"[INFO] Saved JSON file: {scaled_json_path}")

    except Exception as e:
        print(f"[ERROR] Error during processing for {image_file}: {e}")
        if os.path.exists(scaled_image_path):
            os.remove(scaled_image_path)
            print(f"[DEBUG] Deleted scaled image due to error: {scaled_image_path}")


def save_original_image(image_file, class_path, output_folder):
    """
    원본 이미지 및 JSON 저장.
    """
    image_path = os.path.normpath(os.path.join(class_path, image_file))
    json_path = os.path.normpath(os.path.join(class_path, os.path.splitext(image_file)[0] + '.json'))

    print("image_file:", image_file)  # 디버깅 코드
    print("Original JSON path:", json_path)  # 디버깅 코드

    if not os.path.exists(json_path):
        print(f"[WARNING] No JSON file for original image: {image_file}. Skipping...")
        return

    if not os.path.exists(image_path):
        print(f"[ERROR] Image file not found: {image_path}. Skipping...")
        return

    try:
        # 저장할 원본 이미지 경로 (이름 변경 없이 그대로 사용)
        original_image_path = os.path.normpath(os.path.join(output_folder, image_file))
        cv2.imwrite(original_image_path, cv2.imread(image_path))
        print(f"[INFO] Saved original image: {original_image_path}")

        # JSON 파일 경로를 `output_folder` 기준으로 생성
        original_json_name = os.path.splitext(image_file)[0] + ".json"
        original_json_path = os.path.normpath(os.path.join(output_folder, original_json_name))

        # JSON 파일 복사
        with open(json_path, 'r') as jf:
            json_data = json.load(jf)

        # 기존 데이터를 그대로 저장
        with open(original_json_path, 'w') as ojf:
            json.dump(json_data, ojf, indent=4)
        print(f"[INFO] Saved original JSON file: {original_json_path}")

    except Exception as e:
        print(f"[ERROR] Error during saving original for {image_file}: {e}")

# Example usage
input_directory = "C:/Users/Sesame/Desktop/debug"
output_directory_scale50 = "C:/Users/Sesame/Desktop/debug/scaling_output_scale50_org100"
output_directory_scale100 = "C:/Users/Sesame/Desktop/debug/scaling_output_scale100_org50"

coordinates = {
    'rice': (0, 256, 320, 384),  # 하단 왼쪽
    'soup': (320, 256, 320, 384),  # 하단 오른쪽
    'side': [
        (0, 0, 160, 256),    # 상단 1구
        (160, 0, 160, 256),  # 상단 2구
        (320, 0, 160, 256),  # 상단 3구
        (480, 0, 160, 256)   # 상단 4구
    ]
}

class_folders = {
    'rice': ['01011001', '01012006', '01012002'],
    'soup': ['04011005', '04011007', '04017001', '04011011'],
    'side': ['06012004', '07014001', '11013007', '12011008', '06012008', '08011003', '10012001', '11013002', '12011003', '07013003', '11013010']
}

scale_and_save_images_to_separate_folders(
    input_dir=input_directory,
    output_dir_scale50=output_directory_scale50,
    output_dir_scale100=output_directory_scale100,
    coordinates=coordinates,
    class_folders=class_folders,
    seed=42
)
