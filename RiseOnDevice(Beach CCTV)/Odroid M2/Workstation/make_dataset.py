import os

# ==========================================
# 1. 설정 (사용자 환경에 맞게 수정하세요)
# ==========================================
image_dir = './datasets/images/val'  # 이미지 파일들이 들어있는 폴더 경로
output_file = 'dataset.txt'          # 생성될 txt 파일 이름
target_count = 300                   # 추출할 이미지 장수

# 지원하는 이미지 확장자
valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')

# ==========================================
# 2. 이미지 리스트 확보 및 정렬
# ==========================================
# 폴더 내 이미지 파일을 가져와 이름순(시간순)으로 정렬합니다.
files = [f for f in os.listdir(image_dir) if f.lower().endswith(valid_extensions)]
files.sort()

total_files = len(files)
if total_files < target_count:
    print(f"경고: 전체 이미지 수({total_files})가 목표 수({target_count})보다 적습니다. 전체 이미지를 사용합니다.")
    target_count = total_files

# ==========================================
# 3. 일정 간격 추출 및 파일 쓰기
# ==========================================
# 간격(step) 계산
step = total_files / target_count

selected_paths = []
for i in range(target_count):
    # 등간격 인덱스 계산
    idx = int(i * step)
    if idx < total_files:
        # RKNN 툴킷 권장사항인 '절대 경로'로 변환
        abs_path = os.path.abspath(os.path.join(image_dir, files[idx]))
        selected_paths.append(abs_path)

# 텍스트 파일 생성
with open(output_file, 'w') as f:
    for path in selected_paths:
        f.write(path + '\n')

print(f"완료: {total_files}장 중 {len(selected_paths)}장의 경로를 '{output_file}'에 저장했습니다.")
