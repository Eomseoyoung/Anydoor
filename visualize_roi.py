import cv2
import numpy as np
import os

# ROI 설정 (현재 run_inference.py와 동일)
roi_ratio = {
    'y_start': 0.65,
    'y_end': 0.9,
    'x_start': 0.25,
    'x_end': 0.75
}

# 샘플 마스크 파일 로드
mask_folder = './anydoor_dataset/mask/'
mask_files = [f for f in os.listdir(mask_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
sample_mask_path = os.path.join(mask_folder, mask_files[0])

# 마스크 로드
mask_img = cv2.imread(sample_mask_path)
tar_mask = mask_img[:,:,0] > 128
tar_mask = tar_mask.astype(np.uint8)

h, w = tar_mask.shape
y1 = int(h * roi_ratio['y_start'])
y2 = int(h * roi_ratio['y_end'])
x1 = int(w * roi_ratio['x_start'])
x2 = int(w * roi_ratio['x_end'])

# ROI 마스크 생성
roi_mask = np.zeros_like(tar_mask)
roi_mask[y1:y2, x1:x2] = 1

# 교집합 계산
intersection_mask = tar_mask & roi_mask

# 시각화를 위한 컬러 이미지 생성
vis_original = np.zeros((h, w, 3), dtype=np.uint8)
vis_original[tar_mask == 1] = [255, 255, 255]  # 원본 마스크: 흰색

vis_roi = np.zeros((h, w, 3), dtype=np.uint8)
vis_roi[roi_mask == 1] = [0, 255, 0]  # ROI 영역: 녹색

vis_intersection = np.zeros((h, w, 3), dtype=np.uint8)
vis_intersection[intersection_mask == 1] = [0, 0, 255]  # 교집합: 빨강

# 종합 시각화 (모든 정보를 한 이미지에)
vis_combined = np.zeros((h, w, 3), dtype=np.uint8)
vis_combined[tar_mask == 1] = [100, 100, 100]  # 원본 마스크: 회색
vis_combined[roi_mask == 1] = [0, 200, 0]  # ROI 영역: 녹색 (반투명 효과)
vis_combined[intersection_mask == 1] = [0, 0, 255]  # 실제 사용될 영역: 빨강

# ROI 경계선 그리기
cv2.rectangle(vis_combined, (x1, y1), (x2, y2), (0, 255, 255), 3)

# 텍스트 추가
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(vis_combined, f"ROI: ({x1},{y1}) to ({x2},{y2})", (10, 30),
            font, 0.7, (255, 255, 255), 2)
cv2.putText(vis_combined, f"Image size: {w}x{h}", (10, 60),
            font, 0.7, (255, 255, 255), 2)
cv2.putText(vis_combined, "Gray: Original Mask", (10, h-80),
            font, 0.6, (100, 100, 100), 2)
cv2.putText(vis_combined, "Green: ROI Area", (10, h-50),
            font, 0.6, (0, 200, 0), 2)
cv2.putText(vis_combined, "Red: Intersection (Used)", (10, h-20),
            font, 0.6, (0, 0, 255), 2)

# 결과 저장
cv2.imwrite('roi/roi_visualization_original_mask.png', vis_original)
cv2.imwrite('roi/roi_visualization_roi_area.png', vis_roi)
cv2.imwrite('roi/roi_visualization_intersection.png', vis_intersection)
cv2.imwrite('roi/roi_visualization_combined.png', vis_combined)

print(f"샘플 마스크 파일: {mask_files[0]}")
print(f"이미지 크기: {w} x {h}")
print(f"ROI 좌표: ({x1}, {y1}) to ({x2}, {y2})")
print(f"ROI 크기: {x2-x1} x {y2-y1}")
print("\n시각화 이미지 저장 완료:")
print("  - roi_visualization_original_mask.png : 원본 마스크 (흰색)")
print("  - roi_visualization_roi_area.png : ROI 영역 (녹색)")
print("  - roi_visualization_intersection.png : 교집합 - 실제 사용될 영역 (빨강)")
print("  - roi_visualization_combined.png : 종합 시각화 (모든 정보)")
