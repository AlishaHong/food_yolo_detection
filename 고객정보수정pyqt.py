import sys
import pandas as pd
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget, QMessageBox
)

class CustomerWeightUpdater(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # 윈도우 설정
        self.setWindowTitle("고객 몸무게 수정")
        self.setGeometry(100, 100, 400, 300)

        # 레이블과 입력 필드 생성
        self.label_id = QLabel("고객 ID:", self)
        self.input_id = QLineEdit(self)

        self.label_weight = QLabel("새 몸무게 (kg):", self)
        self.input_weight = QLineEdit(self)

        # 업데이트 버튼 생성
        self.btn_update = QPushButton("수정하기", self)
        self.btn_update.clicked.connect(self.update_weight)

        # 종료 버튼 생성
        self.btn_exit = QPushButton("종료", self)
        self.btn_exit.clicked.connect(self.close_application)

        # 레이아웃 설정
        layout = QVBoxLayout()
        layout.addWidget(self.label_id)
        layout.addWidget(self.input_id)
        layout.addWidget(self.label_weight)
        layout.addWidget(self.input_weight)
        layout.addWidget(self.btn_update)
        layout.addWidget(self.btn_exit)

        # 중앙 위젯 설정
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def update_weight(self):
        # 입력값 가져오기
        customer_id = self.input_id.text()
        new_weight = self.input_weight.text()
        csv_path = "FOOD_DB/food_project_user_info.csv"  # 고객 정보가 저장된 CSV 파일 경로

        # 입력값 검증
        if not customer_id or not new_weight:
            QMessageBox.warning(self, "입력 오류", "모든 필드를 입력해주세요.")
            return

        try:
            new_weight = float(new_weight)
        except ValueError:
            QMessageBox.warning(self, "입력 오류", "몸무게는 숫자여야 합니다.")
            return

        # CSV 업데이트 로직 실행
        try:
            # CSV 파일 읽기
            customer_data = pd.read_csv(csv_path)

            if '고객아이디' not in customer_data.columns:
                QMessageBox.critical(self, "CSV 오류", "CSV 파일에 '고객아이디' 열이 없습니다.")
                return

            # 고객 ID로 필터링
            customer_index = customer_data[customer_data['고객아이디'] == customer_id].index

            if not customer_index.empty:
                # 몸무게 수정
                customer_data.loc[customer_index, '몸무게'] = new_weight
                customer_data.to_csv(csv_path, index=False)
                QMessageBox.information(self, "수정 완료", f"고객 ID {customer_id}의 몸무게가 {new_weight}kg으로 수정되었습니다.")
            else:
                QMessageBox.warning(self, "수정 실패", f"고객 ID {customer_id}에 해당하는 정보를 찾을 수 없습니다.")
        except FileNotFoundError:
            QMessageBox.critical(self, "파일 오류", f"CSV 파일 {csv_path}을(를) 찾을 수 없습니다.")
        except Exception as e:
            QMessageBox.critical(self, "오류 발생", f"오류가 발생했습니다: {e}")

    def close_application(self):
        """
        애플리케이션 종료 함수.
        """
        reply = QMessageBox.question(
            self, "종료 확인", "애플리케이션을 종료하시겠습니까?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            QApplication.quit()  # 애플리케이션 종료

# PyQt5 애플리케이션 실행
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CustomerWeightUpdater()
    window.show()
    sys.exit(app.exec_())
