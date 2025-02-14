import os
import shutil
import random

def split_dataset_to_class_folders(base_dir, output_dir, train_ratio=0.8, valid_ratio=0.2, seed=42):
    random.seed(seed)  # 랜덤 시드 설정

    # 큰 train/valid 폴더 생성
    train_images_dir = os.path.join(output_dir, 'train')
    valid_images_dir = os.path.join(output_dir, 'valid')
    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(valid_images_dir, exist_ok=True)

    # base_dir 내 모든 하위 폴더 처리
    for class_folder in os.listdir(base_dir):
        class_path = os.path.join(base_dir, class_folder)
        if not os.path.isdir(class_path):
            continue

        images_dir = os.path.join(class_path, 'images')
        labels_dir = os.path.join(class_path, 'labels')

        if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
            print(f"Missing 'images' or 'labels' directory in {class_path}. Skipping...")
            continue

        supported_extensions = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
        image_files = [f for f in os.listdir(images_dir) if f.endswith(supported_extensions)]
        random.shuffle(image_files)  # 무작위로 파일 리스트 섞기

        # 데이터셋 분할 인덱스 계산
        total_count = len(image_files)
        train_end = int(total_count * train_ratio)
        valid_end = int(total_count * (train_ratio + valid_ratio))

        # 파일 복사 함수
        def copy_files(file_list, target_dir):
            class_target_dir = os.path.join(target_dir, class_folder)
            os.makedirs(os.path.join(class_target_dir, 'images'), exist_ok=True)
            os.makedirs(os.path.join(class_target_dir, 'labels'), exist_ok=True)

            for image_file in file_list:
                # 이미지 파일 복사
                shutil.copy(
                    os.path.join(images_dir, image_file),  # 원본 데이터 경로
                    os.path.join(class_target_dir, 'images', image_file)  # 새로운 경로
                )

                # 이미지 파일과 동일한 이름의 텍스트 파일 복사
                txt_file = os.path.splitext(image_file)[0] + '.txt'  # 확장자를 제외한 파일 이름으로 .txt 생성
                txt_path = os.path.join(labels_dir, txt_file)
                if os.path.exists(txt_path):  # 텍스트 파일이 존재할 경우에만 복사
                    shutil.copy(
                        txt_path,
                        os.path.join(class_target_dir, 'labels', txt_file)
                    )

        # 데이터셋을 train, valid로 나누어 복사
        copy_files(image_files[:train_end], train_images_dir)
        copy_files(image_files[train_end:valid_end], valid_images_dir)

        print(f"Processed class '{class_folder}': Train={train_end}, Valid={valid_end - train_end}")

    print(f"Dataset split completed: {output_dir}/train, {output_dir}/valid")

# 실행 예시(텍스트파일을 먼저 생성 한 후에 진행해야 함)
base_dir = 'C:/Users/Sesame/Desktop/total_100org_50scaled'
output_dir = 'C:/Users/Sesame/Desktop/total_100org_50scaled_splited'
split_dataset_to_class_folders(base_dir, output_dir)