import os
import shutil

# 빈 폴더와 하위 폴더까지 모두 삭제하는 함수
# 최상위만 지우려 시도했더니 폴더가 삭제되지 않았음 
def remove_empty_folders(folder):
    # 폴더가 존재하는지 확인
    if os.path.exists(folder):
        # 하위 폴더를 재귀적으로 순회하며 삭제
        # *** topdowm을 False로 해야 자식폴더부터 위로 올라온다. 기본값은 true(위에서 아래로)
        for root, dirs, files in os.walk(folder, topdown=False):
            for name in files:
                file_path = os.path.join(root, name)
                print(f"파일 삭제: {file_path}")
                os.remove(file_path)  # 파일 삭제

            for name in dirs:
                dir_path = os.path.join(root, name)
                print(f"폴더 삭제: {dir_path}")
                os.rmdir(dir_path)  # 빈 폴더 삭제

        # 최상위 폴더 삭제
        print(f"최상위 폴더 삭제: {folder}")
        os.rmdir(folder)

# 원본 이미지폴더, 원본 라벨폴더, 새로 정리된 베이스폴더 지정
def organize_files_and_remove_empty_folders(base_img_folder, label_folder, new_base_folder):
    # 새로운 image/label 폴더 생성
    image_for_yolo = os.path.join(new_base_folder, "image")
    label_for_yolo = os.path.join(new_base_folder, "label")
    os.makedirs(image_for_yolo, exist_ok=True)
    os.makedirs(label_for_yolo, exist_ok=True)

    # 클래스별로 순회
    for class_name in os.listdir(base_img_folder):
        class_image_path = os.path.join(base_img_folder, class_name)
        class_label_path = os.path.join(label_folder, class_name)

        # 클래스별 폴더 생성 (image와 label)
        class_image_dest = os.path.join(image_for_yolo, class_name)
        class_label_dest = os.path.join(label_for_yolo, class_name)
        os.makedirs(class_image_dest, exist_ok=True)
        os.makedirs(class_label_dest, exist_ok=True)

        # Q1~Q5 폴더에서 이미지 및 라벨 파일 이동
        for q_folder in ["Q1", "Q2", "Q3", "Q4", "Q5"]:
            q_image_path = os.path.join(class_image_path, q_folder)
            q_label_path = os.path.join(class_label_path, q_folder)

            # 이미지 파일 이동
            if os.path.exists(q_image_path):
                for file_name in os.listdir(q_image_path):
                    if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        src_path = os.path.join(q_image_path, file_name)
                        dst_path = os.path.join(class_image_dest, file_name)

                        # 파일명 충돌 방지: _copy 붙이기
                        if os.path.exists(dst_path):
                            base_name, ext = os.path.splitext(file_name)
                            dst_path = os.path.join(class_image_dest, f"{base_name}_copy{ext}")

                        shutil.move(src_path, dst_path)
                        print(f"이미지 이동 완료: {src_path} -> {dst_path}")

                # Q 폴더가 비었으면 삭제
                if not os.listdir(q_image_path):
                    os.rmdir(q_image_path)
                    print(f"{q_image_path} 폴더 삭제 완료")

            # 라벨 파일 이동
            if os.path.exists(q_label_path):
                for txt_file in os.listdir(q_label_path):
                    if txt_file.endswith(".txt"):
                        src_txt_path = os.path.join(q_label_path, txt_file)
                        dst_txt_path = os.path.join(class_label_dest, txt_file)

                        # 파일명 충돌 방지: _copy 붙이기
                        if os.path.exists(dst_txt_path):
                            base_name, ext = os.path.splitext(txt_file)
                            dst_txt_path = os.path.join(class_label_dest, f"{base_name}_copy{ext}")

                        shutil.move(src_txt_path, dst_txt_path)
                        print(f"라벨 이동 완료: {src_txt_path} -> {dst_txt_path}")

                # Q 폴더가 비었으면 삭제
                if not os.listdir(q_label_path):
                    os.rmdir(q_label_path)
                    print(f"{q_label_path} 폴더 삭제 완료")

# 경로 설정
train_image_folder = "C:/Users/SBA/repository/image_detection/음식이미지/Training/[원천]양추정_TRAIN/양추정_이미지_TRAIN/image"
train_label_folder = "C:/Users/SBA/repository/image_detection/음식이미지/Training/[라벨]양추정_TRAIN/txt"
valid_image_folder = "C:/Users/SBA/repository/image_detection/음식이미지/Validation/[원천]양추정_VAL/image"
valid_label_folder = "C:/Users/SBA/repository/image_detection/음식이미지/Validation/[라벨]양추정_VAL/txt"


# 삭제할 빈 폴더 경로 --> xml파일은 아직 지우지 않음 
clean_train_image_folder = "C:/Users/SBA/repository/image_detection/음식이미지/Training/[원천]양추정_TRAIN"
clean_train_label_folder = "C:/Users/SBA/repository/image_detection/음식이미지/Training/[라벨]양추정_TRAIN/txt"
clean_valid_image_folder = "C:/Users/SBA/repository/image_detection/음식이미지/Validation/[원천]양추정_VAL"
clean_valid_label_folder = "C:/Users/SBA/repository/image_detection/음식이미지/Validation/[라벨]양추정_VAL/txt"

# 데이터 정리 및 빈 폴더 삭제 수행(빈폴더 -> q폴더)
organize_files_and_remove_empty_folders(train_image_folder, train_label_folder, "C:/Users/SBA/repository/image_detection/음식이미지/Training")
organize_files_and_remove_empty_folders(valid_image_folder, valid_label_folder, "C:/Users/SBA/repository/image_detection/음식이미지/Validation")

# 빈 폴더 삭제 수행
remove_empty_folders(clean_train_image_folder)
remove_empty_folders(clean_train_label_folder)
remove_empty_folders(clean_valid_image_folder)
remove_empty_folders(clean_valid_label_folder)

print("\n모든 파일이 정리되고 빈 폴더가 삭제되었습니다!")
