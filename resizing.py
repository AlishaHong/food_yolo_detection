import os
import cv2

# basepath 설정
basepath = "C:/Users/Sesame/Desktop/food_train_image/12011003"  # 실제 데이터가 담긴 폴더 경로로 변경해주세요.

# resize 목표 크기
target_size = (640, 640)

# 지원하는 이미지 확장자 목록
image_extensions = ('.jpg', '.jpeg', '.png','.JPG','.JPEG')

# basepath 아래의 모든 파일 순회
for filename in os.listdir(basepath):
    # 이미지 파일 경로
    image_path = os.path.join(basepath, filename)
    
    # 파일이 이미지인지 확인 (확장자 검사)
    if filename.lower().endswith(image_extensions):
        # 이미지 불러오기
        image = cv2.imread(image_path)
        
        # 이미지 읽기 실패 확인
        if image is None:
            print(f"이미지를 읽을 수 없습니다: {filename}")
            continue
        
        # resize 수행
        resized_image = cv2.resize(image, target_size)

        # resize된 이미지를 원본 이미지 파일 경로에 덮어쓰기
        cv2.imwrite(image_path, resized_image)

        print(f"{filename} 이미지 resize 완료.")


print("모든 이미지 resize 완료.")