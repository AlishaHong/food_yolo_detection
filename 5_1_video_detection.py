from ultralytics import YOLO
import cv2
import logging
import os

# 로깅 수준을 WARNING으로 설정하여 불필요한 출력 억제
logging.getLogger().setLevel(logging.WARNING)

# # 모델 경로 리스트


# # 동영상 파일 리스트
# video_paths = [
#     'video_test/black_angle_30.mp4',
#     'video_test/black_angle_60.mp4',
#     'video_test/black_angle_90.mp4',
#     'video_test/white_angle_30.mp4',
#     'video_test/white_angle_60.mp4',
#     'video_test/white_angle_90.mp4',
#     'video_test/wood_angle_30.mp4',
#     'video_test/wood_angle_60.mp4',
#     'video_test/wood_angle_90.mp4'
# ]

model_paths = ['best_pt_정리/9_전체ETC제거_3_4_데이터유지_8_1_total_remove_etc_with_34_1time(8_x_aug_2080)/best.pt',
                'best_pt_정리/9_전체ETC제거_3_4_데이터유지_8_2_total_remove_etc_with_34_3times(8_o_aug_2080)/best.pt',
                'best_pt_정리/10_전체ETC제거_3_4_데이터제거_8_3_total_remove_etc_without_34_1time(8_x_aug_3060)/best.pt',
                'best_pt_정리/10_전체ETC제거_3_4_데이터제거_8_4_total_remove_etc_without_34_3times(8_o_aug_3060)/best.pt']

# model_paths = ['yolo11n.pt']

video_paths = [
    'video_test/black_angle_30.mp4',
    'video_test/black_angle_60.mp4',
    'video_test/black_angle_90.mp4',
    'video_test/white_angle_30.mp4',
    'video_test/white_angle_60.mp4',
    'video_test/white_angle_90.mp4',
    'video_test/wood_angle_30.mp4',
    'video_test/wood_angle_60.mp4',
    'video_test/wood_angle_90.mp4'
]

# 저장된 모델명으로 폴더 이름을 추출하는 함수
def make_folder_name(model_path):
    folder_name = model_path.split('/')[1] + '_result_videos'
    return folder_name

# 동영상 파일 이름 추출 (확장자 제외)
def get_filename_without_extension(filepath):
    return os.path.splitext(os.path.basename(filepath))[0]

# 고유 파일 이름 생성 함수
def get_unique_filename(folder_name, base_name, ext='.avi'):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)  # 폴더가 없으면 생성
    i = 1
    filename = os.path.join(folder_name, f"{base_name}{ext}")   # ext : 파일 확장자
    while os.path.exists(filename):
        filename = os.path.join(folder_name, f"{base_name}_{i}{ext}")
        i += 1
    return filename

# 각 모델에 대해 동영상 처리
for model_path in model_paths:
    # YOLO 모델 로드
    model = YOLO(model_path, verbose=False)
    
    # 폴더 이름 만들기
    folder_name = make_folder_name(model_path)

    for video_path in video_paths:
        # 비디오 파일 불러오기
        video = cv2.VideoCapture(video_path)
        fps = video.get(cv2.CAP_PROP_FPS)  # 실제 FPS 가져오기
        frame_time = int(1000 / fps)  # 프레임 간 시간 (밀리초 단위)

        # 동영상 파일 이름 추출
        base_name = get_filename_without_extension(video_path)

        # 첫 프레임 읽기
        ret, frame = video.read()
        if ret:
            video_filename = get_unique_filename(folder_name, base_name)
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(video_filename, fourcc, fps, (frame.shape[1], frame.shape[0]))
            print(f"모델: {model_path}, 녹화 시작: {video_filename}")
        else:
            print(f"동영상을 읽을 수 없음: {video_path}")
            continue

        # 영상 처리 루프
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break

            # YOLO 모델을 사용하여 객체 감지
            results = model(frame)

            # 감지된 객체들을 프레임에 표시
            annotated_frame = results[0].plot()

            # 프레임을 파일에 저장
            if out is not None:
                out.write(annotated_frame)

            # 실시간 영상 표시
            cv2.imshow('Real-Time Object Detection', annotated_frame)

            # 원래 프레임 시간보다 약간 짧은 대기 시간을 설정하여 더 빠르게 재생
            key = cv2.waitKey(int(frame_time * 0.8))  # 원래 속도보다 약 20% 빠르게 재생

            if key == 27:  # ESC 키를 누르면 종료
                break

        # 녹화 종료 및 리소스 정리
        video.release()
        if out is not None:
            out.release()
            print(f"모델: {model_path}, 녹화 종료: {video_filename}")

cv2.destroyAllWindows()
