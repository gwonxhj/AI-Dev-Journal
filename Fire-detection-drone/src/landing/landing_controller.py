import cv2
import numpy as np
import time
from pipeline import gstreamer_pipeline
from pymavlink import mavutil

def gstreamer_pipeline(
    sensor_id,
    capture_width=1280,
    capture_height=720,
    display_width=960,
    display_height=540,
    framerate=30,
    flip_method=0,
):
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width=(int){capture_width}, height=(int){capture_height}, framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
    )

class Landing:
    def __init__(self, master, display_width=960, display_height=540):
        self.frame_width = display_width
        self.frame_height = display_height
        self.last_print_time = 0
        self.master = master  # MAVLink 연결 객체
        self.last_command = [0, 0, 0, 0]

        self.state = "SEARCHING"  # "SEARCHING" 또는 "MATCHING"
        self.search_altitudes = [5.0, 4.0, 3.0, 2.0]  # 미터 단위
        self.target_altitude_index = 0
        self.current_altitude = self.search_altitudes[self.target_altitude_index]
        self.matched_position = None

        self.target_x = self.frame_width // 2
        self.target_y = (self.frame_height // 2 + self.frame_height) // 2
        
        self.SCALE = 15  # 정수형으로 변환하기 위한 스케일링 팩터

    def detect_red_dot(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 150, 150])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 150, 150])
        upper_red2 = np.array([179, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)
        
        kernel = np.ones((3, 3), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            approx = cv2.approxPolyDP(largest_contour, 0.02 * cv2.arcLength(largest_contour, True), True)
            
            if area > 150 and len(approx) > 7:
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    return (cx, cy, area)
        return None

    def _get_altitude_from_fc(self):
        """MAVLink 메시지로부터 고도값을 가져오는 함수"""
        try:
            msg = self.master.recv_match(type=['GLOBAL_POSITION_INT'], blocking=False, timeout=0.1)
            if msg and msg.get_type() == 'GLOBAL_POSITION_INT':
                # relative_alt는 mm 단위이므로 m 단위로 변환
                return msg.relative_alt / 1000.0
        except Exception as e:
            print(f"FC에서 고도값을 읽는 중 오류 발생: {e}")
        return None

    def get_control_command(self, frame):
        marker_info = self.detect_red_dot(frame)
        
        cv2.line(frame, (0, self.target_y), (self.frame_width, self.target_y), (0, 255, 255), 2)
        cv2.line(frame, (self.target_x, 0), (self.target_x, self.frame_height), (0, 255, 255), 2)
        
        if self.state == "SEARCHING":
            if marker_info:
                print("마커 감지! 매칭 모드로 전환합니다.")
                self.state = "MATCHING"
                self.matched_position = (marker_info[0], marker_info[1])
                return self._get_matching_command(frame, marker_info)
            else:
                return self._get_search_command()

        elif self.state == "MATCHING":
            if marker_info:
                return self._get_matching_command(frame, marker_info)
            else:
                print("매칭 중 마커 손실. 탐색 모드로 돌아갑니다.")
                self.state = "SEARCHING"
                return [0, 0, 0, 0]

    def _get_search_command(self):
        current_altitude_from_fc = self._get_altitude_from_fc()

        if self.target_altitude_index < len(self.search_altitudes):
            target_altitude = self.search_altitudes[self.target_altitude_index]
            
            if current_altitude_from_fc is None or current_altitude_from_fc > target_altitude + 0.5:
                print(f"마커 없음. 고도를 {target_altitude}m로 낮춥니다.")
                return [-0.5, 0, 0, 0] # 고도를 낮추는 명령
            else:
                self.target_altitude_index += 1
                if self.target_altitude_index >= len(self.search_altitudes):
                    print("마커를 찾지 못했습니다. 착륙 실패.")
                    return [0, 0, 0, 0]
                else:
                    print(f"현재 고도 {target_altitude}m에 도달. 다음 목표 고도 {self.search_altitudes[self.target_altitude_index]}m로 진행")
                    return [0, 0, 0, 0] # 다음 고도까지 호버링

        print("마커를 찾지 못했습니다. 착륙 실패.")
        return [0, 0, 0, 0]

    def _get_matching_command(self, frame, marker_info):
        marker_center_x, marker_center_y, _ = marker_info
        
        current_altitude_from_fc = self._get_altitude_from_fc()
        LANDING_ALTITUDE_THRESHOLD = 0.3  # 0.3m (30cm)

        if current_altitude_from_fc and current_altitude_from_fc <= LANDING_ALTITUDE_THRESHOLD:
            print("착륙 완료!")
            return [0, 0, 0, 0] # 모든 명령 정지
        
        x_diff = marker_center_x - self.target_x
        y_diff = marker_center_y - self.target_y
        
        command_down = -0.2
        command_forward_backward = -(y_diff // self.SCALE)
        command_left_right = -(x_diff // self.SCALE)
        
        now = time.time()
        if now - self.last_print_time > 0.5:
            altitude_str = f"{current_altitude_from_fc:.2f}m" if current_altitude_from_fc is not None else "N/A"
            print(f"매칭 중 | 고도: {altitude_str} | 오차: x={x_diff}, y={y_diff} | 명령: [{command_down}, {command_forward_backward}, {command_left_right}, 0]")
            self.last_print_time = now
            
        return [command_down, command_forward_backward, command_left_right, 0]

    def run_main_loop(self):
        master = mavutil.mavlink_connection('udp:127.0.0.1:14550', baud=57600)
        master.wait_heartbeat()
        print("하트비트 수신 성공, 드론 연결 완료!")

        self.master = master # Landing에 MAVLink 연결 할당

        pipeline0 = gstreamer_pipeline(sensor_id=0)
        cap = cv2.VideoCapture(pipeline0, cv2.CAP_GSTREAMER)

        if not cap.isOpened():
            print("GStreamer 파이프라인을 열 수 없습니다. 카메라 연결이나 설정을 확인하세요.")
            return

        print("카메라를 시작합니다. 'ESC' 키를 눌러 종료하세요.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("프레임을 읽을 수 없습니다.")
                break
                
            command = self.get_control_command(frame)
            
            cv2.imshow("Drone Landing", frame)
            
            if cv2.waitKey(1) & 0xFF == 27:
                print("사용자 중지 요청.")
                break
        
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    landing_controller = Landing(master=None)
    landing_controller.run_main_loop()
