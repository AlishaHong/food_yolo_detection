import os
import glob
import cv2
import sys

class ImageProcessor:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.file_names = self.get_image_list()

    # jpg 원본 리스트 불러오기
    def get_image_list(self):
        data_org = os.path.join(self.data_dir, 'snack_dataOrg')
        file_names = glob.glob(os.path.join(data_org, '*.jpg'))
        return file_names

    # 원본 이미지 읽기
    def load_image_by_name(self, img_name):
        for file_name in self.file_names:
            if img_name in file_name:
                print(f"Loading image: {file_name}")
                img = cv2.imread(file_name)
                if img is None:
                    sys.exit('Image load failed')
                return img
        sys.exit(f"{img_name} 파일이 없어요!")

    # 리사이징한 이미지를 저장할 저장 경로 생성 함수
    def make_save_path(self, img_name):
        data_path = os.path.join(self.data_dir, 'snack_dataOrg_640')
        if not os.path.exists(data_path):
            os.makedirs(data_path)

        save_file_name = os.path.join(data_path, f'{img_name}.jpg')
        return save_file_name

    # 이미지 저장 함수
    def save_image(self, img, img_name):
        save_path = self.make_save_path(img_name)
        cv2.imwrite(save_path, img)
        print(f"Image saved to: {save_path}")

    # 이미지 사이즈 조절하기
    def resize_image640(self, img, img_name, width=640, height=640):
        resized_img = cv2.resize(img, (width, height))
        self.save_image(resized_img, img_name)

    # 전체 이미지 처리 함수
    def process_images(self):
        for file_name in self.file_names:
            basename = os.path.basename(file_name)
            img_name, _ = os.path.splitext(basename)
            img = self.load_image_by_name(img_name)
            self.resize_image640(img, img_name)

    # snack_dataOrg_640 폴더에 있는 이미지 크기 출력(확인용)
    def print_image_sizes(self):
        data_path = os.path.join(self.data_dir, 'snack_dataOrg_640')
        resized_file_names = glob.glob(os.path.join(data_path, '*.jpg'))
        
        for file_name in resized_file_names:
            img = cv2.imread(file_name)
            if img is not None:
                height, width, _ = img.shape
                print(f"{file_name}: {width}x{height}")
            else:
                print(f"Failed to load image: {file_name}")

# 메인 실행 함수
def main():
    data_dir = os.getcwd()
    processor = ImageProcessor(data_dir)  # 객체 생성
    processor.process_images()  # 전체 이미지 처리
    processor.print_image_sizes()  # 변환된 이미지 크기 출력(확인용)

if __name__ == "__main__":
    main()
