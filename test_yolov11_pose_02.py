import cv2
import time
from ultralytics import YOLO

print('Starting...')

model = YOLO('yolo11n-pose.pt')  # YOLOv11 포즈 모델 경로

# 동영상 파일 사용 시
# video_path = 'test.mp4'
# cap = cv2.VideoCapture(video_path)

# webcam 사용 시
cap = cv2.VideoCapture(0)

# 초기 화면 크기 설정
initial_width, initial_height = 640, 480
cv2.namedWindow("YOLOv11 Inference", cv2.WINDOW_NORMAL)  # 윈도우가 자유롭게 크기 변경 가능
cv2.resizeWindow("YOLOv11 Inference", initial_width, initial_height)

# 사용자 입력: 추론 시간 간격 설정 (초)
time_interval = input("추론 간격을 설정하세요 (초 단위, 실시간 추론은 엔터): ")

# 시간 간격 설정이 없는 경우 실시간 추론
if time_interval.strip() == "":
    time_interval = None
else:
    time_interval = float(time_interval)

start_time = time.time()
results = None
count = 0

while cap.isOpened():
    # 비디오에서 프레임 읽기
    success, frame = cap.read()

    count += 1

    if success:
        # 현재 윈도우 크기를 가져와 그 크기에 맞게 프레임 크기 조정
        current_width = cv2.getWindowImageRect("YOLOv11 Inference")[2]
        current_height = cv2.getWindowImageRect("YOLOv11 Inference")[3]
        frame = cv2.resize(frame, (current_width, current_height))

        # 시간 간격에 따른 추론 실행
        if time_interval is None or (time.time() - start_time) > time_interval:
            results = model(frame)
            start_time = time.time()  # 타이머 갱신

        if results is not None:
            # 시각화된 결과를 윈도우에 표시
            annotated_frame = results[0].plot()
            cv2.imshow("YOLOv11 Inference", annotated_frame)

            P = 0
            try:
                for idx, kpt in enumerate(results[0].keypoints[0]):
                    print('Persons Detected')
                    P = 1
            except:
                print('No Persons')
                P = 0

        # 'q' 키를 누르면 루프 종료
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

# 캡처 및 윈도우 해제
cap.release()
cv2.destroyAllWindows()

