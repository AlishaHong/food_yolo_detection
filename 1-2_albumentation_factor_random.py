import os
import cv2
from matplotlib import pyplot as plt
import albumentations as A
import random

def yolo_to_coco(img_width, img_height, yolo_bbox):
    """
    YOLO 형식의 bounding box를 COCO 형식으로 변환합니다.
    """
    x_center, y_center, width, height = yolo_bbox
    x_min = int((x_center - width / 2) * img_width)
    y_min = int((y_center - height / 2) * img_height)
    width = int(width * img_width)
    height = int(height * img_height)
    return [x_min, y_min, width, height]


def visualize_bbox(img, yolo_bbox, class_name, color=(255, 0, 0), thickness=2):
    # 단일 바운딩 박스 시각화
    img_width = img.shape[1]
    img_height = img.shape[0]
    bbox = yolo_to_coco(img_width, img_height, yolo_bbox)
    x_min, y_min, w, h = bbox
    x_max = x_min + w
    y_max = y_min + h

    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), color, -1)
    cv2.putText(img, class_name, (x_min, y_min - int(0.3 * text_height)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), lineType=cv2.LINE_AA)
    return img


def visualize(image, bboxes, category_ids, category_id_to_name):
    """
    이미지에 여러 개의 bounding box를 시각화합니다.
    """
    img = image.copy()
    for bbox, category_id in zip(bboxes, category_ids):
        class_name = category_id_to_name[category_id]
        img = visualize_bbox(img, bbox, class_name)
    plt.figure(figsize=(12, 12))
    plt.axis('off')
    plt.imshow(img)
    plt.show()


def read_bounding_boxes(file_path):
    # txt 파일에서 바운딩박스와 클래스번호 읽어온 뒤 리스트에 담아주기
    category_ids = []
    bboxes = []
    with open(file_path, 'r') as f:
        for line in f:
            ids, xc, yc, w, h = line.split(' ')
            category_ids.append(int(ids))
            bboxes.append([float(xc), float(yc), float(w), float(h)])
    return bboxes, category_ids



def apply_random_transformations(image, bboxes, category_ids):
    """
    이미지와 바운딩 박스에 랜덤한 전처리를 적용하고, 변환 설명을 반환합니다.
    """
    # 변환 설명을 저장할 리스트
    transform_description = []

    # 랜덤한 전처리 변환 설정
    rotate_angle = random.randint(-90, 90)  # 회전 각도 설정
    # brightness_limit = random.uniform(0.2, 0.7)  # 밝기 범위 설정  -> 기본값 범위로 하기로함 -0.2~0.2 
    # contrast_limit = random.uniform(0.2, 0.7)  # 대비 범위 설정   -> 기본값 범위로 하기로함 -0.2~0.2 
    gauss_var_limit = random.uniform(1000.0, 4000.0)  # 가우스 노이즈 설정

    transforms = A.Compose([
        A.RandomCrop(width = 600, height = 600, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=(rotate_angle, rotate_angle), p=0.8, rotate_method='ellipse'),
        # A.RandomBrightnessContrast(brightness_limit=(brightness_limit, brightness_limit),
        #                            contrast_limit=(contrast_limit, contrast_limit), p=0.2),
        A.RandomBrightnessContrast(p=0.2),
        A.GaussNoise(var_limit=(gauss_var_limit, gauss_var_limit), p=0.3),
        A.MotionBlur(blur_limit = 9, p = 0.7)
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

    # 변환 적용
    transformed = transforms(image=image, bboxes=bboxes, class_labels=category_ids)

    return transformed['image'], transformed['bboxes'], transformed['class_labels'], "_".join(transform_description)


def process_images_in_folder_random(folder_path, output_folder, num_variations=400):
    """
    지정한 폴더의 모든 파일을 읽어와 지정된 횟수만큼 랜덤한 전처리를 적용하고, 변환된 이미지를 저장합니다.
    """
    # class_id = {0: 'hush'}
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg'):
            image_path = os.path.join(folder_path, filename)
            txt_path = image_path.replace('.jpg', '.txt')

            if not os.path.exists(txt_path):
                continue

            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            bboxes, category_ids = read_bounding_boxes(txt_path)
            print(category_ids)

            # 지정된 횟수만큼 랜덤한 변환 적용
            for i in range(num_variations):
                transformed_image, transformed_bboxes, transformed_class_labels, applied_transform = apply_random_transformations(
                    image, bboxes, category_ids
                )

                # 변환 설명이 비어 있는 경우 기본 설명 추가  -> 변환설명은 뺐는데 파일명에 증가되는 숫자를 추가하기로함
                if not applied_transform:
                    applied_transform = f"{i+1}"

                # 새롭게 생성된 파일명 설정 (원본 파일명 + 변환 설명)
                base_filename = os.path.splitext(filename)[0]
                new_image_name = f"{base_filename}_{applied_transform}.jpg"
                new_txt_name = f"{base_filename}_{applied_transform}.txt"

                # 결과 저장
                transformed_image = cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(output_folder, new_image_name), transformed_image)

                with open(os.path.join(output_folder, new_txt_name), 'w') as f:

                    for bbox, class_id in zip(transformed_bboxes, transformed_class_labels):
                        class_id = int(class_id)
                        bbox_str = ' '.join(map(str, bbox))
                        f.write(f"{class_id} {bbox_str}\n")
             

                print(f"파일 저장 완료: {new_image_name}, {new_txt_name}\n")

# 예제 사용
input_folder = 'snack_dataOrg_640'  # 원본 이미지와 TXT 파일이 있는 폴더
output_folder = 'snack_dataAugmentation_random'  # 변환된 이미지와 TXT 파일을 저장할 폴더

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

process_images_in_folder_random(input_folder, output_folder, num_variations=400)




