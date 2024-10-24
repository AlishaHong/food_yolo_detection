import os

def rename_images_to_lower(folder):
    # 폴더 및 하위 폴더 순회
    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith(".JPG"):  # 대문자 확장자만 찾기
                src = os.path.join(root, file)
                dst = os.path.join(root, file[:-4] + ".jpg")  # 확장자 변경
                os.rename(src, dst)
                print(f"{src} -> {dst} 로 변경 완료")

# Training 이미지 폴더 확장자 변경
train_image_folder = "C:/Users/SBA/repository/image_detection/galbitang/Training\image"
rename_images_to_lower(train_image_folder)

# Validation 이미지 폴더 확장자 변경
valid_image_folder = "C:/Users/SBA/repository/image_detection/galbitang/Validation/image"
rename_images_to_lower(valid_image_folder)

print("\n모든 JPG 파일이 jpg로 변경되었습니다!")