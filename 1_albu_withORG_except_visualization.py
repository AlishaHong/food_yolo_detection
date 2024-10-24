import os
import cv2
import albumentations as A
import random

# 바운딩박스 텍스트 파일의 클래스인덱스 읽어옴
def read_bounding_boxes(file_path):
    category_ids = []
    bboxes = []
    with open(file_path, 'r') as f:
        for line in f:
            ids, xc, yc, w, h = line.split(' ')
            category_ids.append(int(ids))
            bboxes.append([float(xc), float(yc), float(w), float(h)])
    return bboxes, category_ids

# 랜덤한 변환을 적용하는 함수
def apply_random_transformations(image, bboxes, category_ids):
    rotate_angle = random.randint(-90, 90)  
    gauss_var_limit = random.uniform(1000.0, 4000.0)  

    transforms = A.Compose([
        A.RandomCrop(width=600, height=600, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=(rotate_angle, rotate_angle), p=0.8, rotate_method='ellipse'),
        A.RandomBrightnessContrast(p=0.2),
        A.GaussNoise(var_limit=(gauss_var_limit, gauss_var_limit), p=0.3),
        A.MotionBlur(blur_limit=9, p=0.7)
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.2))

    transformed = transforms(image=image, bboxes=bboxes, class_labels=category_ids)

    return transformed['image'], transformed['bboxes'], transformed['class_labels']

# 이미지를 증식하고 저장하는 함수
def process_images_in_folder_random(image_root_folder, label_root_folder, num_variations=400):
    # image_root_folder와 label_root_folder의 모든 하위 폴더를 탐색
    for root, _, files in os.walk(image_root_folder):
        for filename in files:
            if filename.endswith('.JPG'):
                # 이미지 파일 경로
                image_path = os.path.join(root, filename)
                
                # 현재 이미지에 대응하는 레이블 파일 경로 생성
                relative_path = os.path.relpath(root, image_root_folder)
                label_path = os.path.join(label_root_folder, relative_path, filename.replace('.JPG', '.txt'))
                
                if not os.path.exists(label_path):
                    print(f"라벨 파일이 없습니다: {label_path}")
                    continue

                # 이미지 읽기
                image = cv2.imread(image_path)
                if image is None:
                    print(f"이미지를 불러올 수 없습니다: {image_path}")
                    continue  # 이미지 로드 실패 시 다음 파일로 넘어감
                
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # 바운딩 박스와 카테고리 ID 읽기
                bboxes, category_ids = read_bounding_boxes(label_path)

                for i in range(num_variations):
                    # 변환 적용
                    transformed_image, transformed_bboxes, transformed_class_labels = apply_random_transformations(
                        image, bboxes, category_ids
                    )

                    # 변환된 파일명 생성
                    base_filename = os.path.splitext(filename)[0]
                    new_image_name = f"{base_filename}_{i+1}.JPG"
                    new_txt_name = f"{base_filename}_{i+1}.txt"

                    # 변환된 이미지 및 레이블 파일 저장
                    transformed_image = cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR)
                    output_image_path = os.path.join(root, new_image_name)
                    output_label_path = os.path.join(label_root_folder, relative_path, new_txt_name)

                    cv2.imwrite(output_image_path, transformed_image)

                    with open(output_label_path, 'w') as f:
                        for bbox, class_id in zip(transformed_bboxes, transformed_class_labels):
                            class_id = int(class_id)
                            bbox_str = ' '.join(map(str, bbox))
                            f.write(f"{class_id} {bbox_str}\n")

                    print(f"파일 저장 완료: {new_image_name}, {new_txt_name}\n")

# 예제 사용
image_root_folder = 'galbitang/Training/image'  # 이미지 클래스들이 있는 최상위 폴더
label_root_folder = 'galbitang/Training/label'  # 라벨 클래스들이 있는 최상위 폴더

process_images_in_folder_random(image_root_folder, label_root_folder, num_variations=2)
