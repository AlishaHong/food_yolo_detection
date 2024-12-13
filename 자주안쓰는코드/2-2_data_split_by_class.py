import os
import shutil
import random

def split_dataset(data_dir, output_dir, train_ratio=0.75, valid_ratio=0.1):
    # 데이터 폴더 내 하위폴더까지 들어가서 모든 jpg 파일 목록 가져옴 
    for root, subdirs, files in os.walk(data_dir):
        for subdir in subdirs:
            # 하위폴더(클래스 폴더) 경로
            subdir_path = os.path.join(root, subdir)
            image_files = [f for f in os.listdir(subdir_path) if f.endswith('.jpg')]
            random.shuffle(image_files)  # 무작위로 섞어줌

            # 데이터셋 분할 갯수 계산
            total_count = len(image_files)
            train_end = int(total_count * train_ratio)
            valid_end = int(total_count * (train_ratio + valid_ratio))

            # 폴더 경로 생성하는데 기존의 코드와 다른점은 images/labels 아래에 클래스 경로를 붙여서 생성해줌
            # for subset in ['train', 'valid', 'test']:
            #     os.makedirs(os.path.join(output_dir, subset, 'images', subdir), exist_ok=True)
            #     os.makedirs(os.path.join(output_dir, subset, 'labels', subdir), exist_ok=True)
            
            for subset in ['train', 'valid']:
                os.makedirs(os.path.join(output_dir, subset, 'images', subdir), exist_ok=True)
                os.makedirs(os.path.join(output_dir, subset, 'labels', subdir), exist_ok=True)

            # 파일 복사
            def copy_files(file_list, subset):
                for image_file in file_list:
                    # 원본경로는 같지만, 이동할 곳 경로에 하위폴더(클래스폴더) 붙여줌
                    shutil.copy(
                        os.path.join(subdir_path, image_file),
                        os.path.join(output_dir, subset, 'images', subdir, image_file)
                    )

                    # 이미지 파일과 동일한 이름의 텍스트 파일 복사
                    txt_file = image_file.replace('.jpg', '.txt')
                    txt_path = os.path.join(subdir_path, txt_file)
                    if os.path.exists(txt_path):  # 텍스트 파일이 존재할 경우에만 복사
                        shutil.copy(
                            txt_path,
                            os.path.join(output_dir, subset, 'labels', subdir, txt_file)
                        )

            # Split the dataset into train, valid, and test sets
            copy_files(image_files[:train_end], 'train')
            copy_files(image_files[train_end:valid_end], 'valid')
            # copy_files(image_files[valid_end:], 'test')

            print(f"{subdir}폴더 분리완료")

    print("분리가 완료되었습니다.")

    # 디렉토리 별 파일 개수 계산
    def count_files_in_directory(directory):
        return sum([len(files) for _, _, files in os.walk(directory)])

    # 디렉토리 별 파일 갯수 출력
    print(f"Train folder image count: {count_files_in_directory(os.path.join(output_dir, 'train', 'images'))}")
    print(f"Train folder label count: {count_files_in_directory(os.path.join(output_dir, 'train', 'labels'))}")
    print(f"Valid folder image count: {count_files_in_directory(os.path.join(output_dir, 'valid', 'images'))}")
    print(f"Valid folder label count: {count_files_in_directory(os.path.join(output_dir, 'valid', 'labels'))}")
    # print(f"Test folder image count: {count_files_in_directory(os.path.join(output_dir, 'test', 'images'))}")
    # print(f"Test folder label count: {count_files_in_directory(os.path.join(output_dir, 'test', 'labels'))}")

# go! 
data_dir = 'origin'  # 원본 데이터 경로
output_dir = 'origin_albu_test'  # 분리한 데이터를 저장할 폴더
split_dataset(data_dir, output_dir)
