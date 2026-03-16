# Python 3.12
import socket
import struct
import numpy as np
import cv2
from collections import deque
import threading
import math
import time

HOST = '0.0.0.0'
PORT = 5000

EXPECTED_W = 256
EXPECTED_H = 256

# ------------------- 큐 -------------------
raw_queue = deque(maxlen=1)
processed_queue = deque(maxlen=1)
gyro_queue = deque(maxlen=1000)# 추후 프로젝트로 인하여 추가된 부분, 여기서 안쓰인다.
accel_queue = deque(maxlen=1000)# 추후 프로젝트로 인하여 추가된 부분, 여기서 안쓰인다.
flow_queue_1 = deque(maxlen=1)
flow_queue_2 = deque(maxlen=1)
roi_queue = deque(maxlen=1)

# ------------------- Depth 노멀라이즈 -------------------
def normalize_depth(depth_frame):
    depth_frame = np.nan_to_num(depth_frame, nan=0.0, posinf=0.0, neginf=0.0)
    min_val = np.min(depth_frame)
    max_val = np.max(depth_frame)
    depth_norm = (depth_frame - min_val) / (max_val - min_val + 1e-6)
    return (depth_norm * 255).astype(np.uint8)

# ------------------- 가중치 계산 -------------------
def compute_depth_confidence(depth_small):
    depth_f = depth_small.astype(np.float32)

    # 1. gradient 계산
    grad_x = cv2.Sobel(depth_f, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(depth_f, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(grad_x, grad_y)

    # 2. 블록 단위로 downsample → 공간적 분포 파악 목적
    # 16x16 정도면 충분 (전체가 256x256이라면 16배 축소)
    block = cv2.resize(grad_mag, (16, 16), interpolation=cv2.INTER_AREA)

    # 3. 블록들의 분산 계산
    spatial_var = block.var()  # 하나의 수

    # 4. sigmoid로 0~1 스케일로 압축 (필요하면)
    # C 값은 조정 가능. variance가 0.0~0.01 정도 나온다면 C=0.005 정도
    C = 0.005
    confidence = 1.0 - math.exp(-spatial_var / C)

    return float(np.clip(confidence, 0.0, 1.0))


def compute_flow_weight(flow_mag_map, k=10.0, x0=0.5):
    """
    flow_mag_map : optical flow magnitude map
    k : sigmoid 기울기, 작게 하면 완만
    x0 : sigmoid 중앙값, W=0.5가 되는 std 기준
    """
    std = float(flow_mag_map.std())
    W = 1.0 / (1.0 + np.exp(-k * (std - x0)))
    return np.clip(W, 0.0, 1.0)

def compute_static_weight(gray_frame, NX, NY):
    gray_umat = cv2.UMat(gray_frame)
    small = cv2.resize(gray_umat, (NX//4, NY//4), interpolation=cv2.INTER_AREA).get()
    std = float(small.std())
    C = 50.0
    W = 1.0 - math.exp(-std / C)
    return np.clip(W, 0.0, 1.0)

#-------------------- 필터링 행렬 -----------------
def compute_depth_matrix(depth_uint8, threshold=0.7, k=20.0):
    # 1) 0~1 정규화
    depth_norm = depth_uint8.astype(np.float32) / 255.0

    # 2) 시그모이드 함수 적용
    #    threshold에서 0.5가 되도록 중심을 이동
    #    k는 기울기(가파름)
    W_depth = 1.0 / (1.0 + np.exp(-k * (depth_norm - threshold)))

    return W_depth

def compute_flow_matrix(flow_norm, f_wflow, k=20.0):
    """
    flow_norm: 이미 0~1로 정규화된 optical flow magnitude 값
    f_wflow : 유효값(= sigmoid의 중앙값, 출력이 0.5가 되는 지점)
    k : 기울기 조절 상수 (20~30 권장)
    """
    flow = np.clip(flow_norm - f_wflow,0.0,1.0)
    
    W_flow = 1.0 / (1.0 + np.exp(-k * (flow-0.5)))
    
    return W_flow

# ------------------- Flow 정규화 기반 신뢰도 -------------------
def compute_flow_score_safe(W_flow, Max_flow, mask, window_size=3, area_thresh=200):
    """
    W_flow: Optical Flow magnitude map, 0~1
    Max_flow: 현재 frame에서 유효 벡터의 최대 이동량
    mask: optical flow 스레드에서 보내는 유효 vector mask (0/255)
    window_size: LK optical flow 계산에 사용한 window step
    area_thresh: 최소 유효 영역 크기(pixel) → 이보다 작으면 정적 환경으로 판단

    반환:
        f_wflow: sigmoid 중심값으로 사용할 유효값 (0~1)
                 정적 환경이면 무조건 1
    """
    # --- 정적 환경 판단 ---
    active_pixels = np.count_nonzero(mask)
    is_dynamic = active_pixels >= area_thresh

    # --- 의미 있는 최대 벡터값 ---
    max_val = 1.15 * window_size

    if not is_dynamic:
        # 정적 환경 → f_wflow 무조건 1
        f_wflow = 1.0
    else:
        # 동적 환경 → 기존 로직 사용, Max_flow가 너무 작으면 1로 보정
        if max_val > Max_flow:
            f_wflow = 1.0
        else:
            f_wflow = max_val / Max_flow

    # f_wflow가 0~1 범위를 넘지 않도록 클립
    f_wflow = np.clip(f_wflow, 0.0, 1.0)
    return f_wflow

# ------------------edge 연결 함수 ----------------
def fill_edges(mask):
    # mask: numpy uint8 binary (0/255)
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_filled = np.zeros_like(mask)
    if contours:
        cv2.drawContours(mask_filled, contours, -1, 255, thickness=cv2.FILLED)
    return mask_filled

# ------------------- TCP 수신 -------------------
def recv_all(sock, n):
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data

def receive_thread(sock, raw_queue, gyro_queue, accel_queue):
    """
    Packet format
    ------------------------------------------------
    [1B] packet_type
      0x01 : frame packet
        [4B] jpeg_len
        [4B] depth_len
        [jpeg bytes]
        [depth bytes]

      0x02 : gyro packet
        [8B] timestamp (ns)
        [4B] gyro_z (rad/s)

      0x03 : accel packet
        [8B] timestamp (ns)
        [4B] accel_x (m/s^2)
        [4B] accel_y (m/s^2)
        [4B] accel_z (m/s^2)

    ------------------------------------------------
    """

    while True:
        # -----------------------------
        # 1) packet type
        # -----------------------------
        pkt_type_raw = recv_all(sock, 1)
        if not pkt_type_raw:
            print("[RECV] connection closed")
            break

        pkt_type = pkt_type_raw[0]

        # -----------------------------
        # 2) FRAME PACKET
        # -----------------------------
        if pkt_type == 0x01:
            header = recv_all(sock, 8)
            if not header:
                break

            jpeg_len, depth_len = struct.unpack('!II', header)

            jpeg_bytes = recv_all(sock, jpeg_len)
            depth_bytes = recv_all(sock, depth_len)

            if jpeg_bytes is None or depth_bytes is None:
                print("[RECV] frame incomplete")
                break

            raw_queue.append(
                (jpeg_bytes, depth_bytes)
            )

        # -----------------------------
        # 3) GYRO PACKET  (추후 프로젝트로 인하여 추가된 부분, 여기서 안쓰인다.)
        # -----------------------------
        elif pkt_type == 0x02:
            payload = recv_all(sock, 12)
            if not payload:
                break

            timestamp, gyro_z = struct.unpack('!qf', payload)

            #print(f"Received gyro: timestamp={timestamp}, gyroZ={gyro_z}")

            gyro_queue.append(
                (timestamp, gyro_z)
            )

        # ------------------------------
        # 4) ACCEL PACKET
        # ------------------------------
        elif pkt_type == 0x03:
            payload = recv_all(sock, 20)
            if not payload:
                break

            timestamp, ax, ay, az = struct.unpack('!qfff', payload)

            pc_ts = time.monotonic_ns()

            #print(f"Received accel: timestamp={timestamp}, ax={ax}, ay={ay}, az={az}")

            accel_queue.append(
                (timestamp, pc_ts, ax, ay, az)
            )


        else:
            print(f"[RECV] unknown packet type: {pkt_type}")
            break

# ------------------- Depth + Saliency 통합 스레드 (최신 frame만) -------------------
def depth_saliency_thread():
    prev_depth = None
    depth_alpha = 0.6
    smoothing_skip = 2
    frame_counter = 0

    while True:
        if not raw_queue:
            time.sleep(0.001)
            continue

        # 항상 최신 frame만
        jpeg_bytes, depth_bytes = raw_queue.pop()
        raw_queue.clear()

        depth_array = np.frombuffer(depth_bytes, dtype='<f4').copy()
        if depth_array.size != EXPECTED_W * EXPECTED_H:
            continue
        depth_frame = depth_array.reshape((EXPECTED_H, EXPECTED_W))

        # Depth downsample + smoothing
        depth_small = cv2.resize(depth_frame, (64, 64), interpolation=cv2.INTER_AREA)
        frame_counter += 1
        if prev_depth is None:
            smoothed = depth_small.copy()
        else:
            if frame_counter % smoothing_skip == 0:
                smoothed = cv2.addWeighted(prev_depth.astype(np.float32),
                                           depth_alpha,
                                           depth_small.astype(np.float32),
                                           1 - depth_alpha, 0.0)
            else:
                smoothed = depth_small.copy()
        prev_depth = smoothed.copy()

        # Depth 정규화
        min_val = np.min(smoothed)
        max_val = np.max(smoothed)
        range_val = max(max_val - min_val, 1e-6)
        depth_uint8 = ((smoothed - min_val)/range_val * 255).astype(np.uint8)

        # =================== Saliency ===================
        cam_array = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        if cam_array is None or cam_array.size == 0:
            print("Empty frame received, skipping...")
            continue
        cam_frame = cv2.imdecode(cam_array, cv2.IMREAD_COLOR)
        if cam_frame is None:
            continue

        gray = cv2.cvtColor(cam_frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (32,32), interpolation=cv2.INTER_LINEAR)
        edges = cv2.Canny(small, 40, 120)
        mask_internal = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)
        mask_internal = cv2.GaussianBlur(mask_internal.astype(np.float32), (5,5), 0)
        _, mask_internal = cv2.threshold(mask_internal, 25, 255, cv2.THRESH_BINARY)
        mask_internal = mask_internal.astype(np.uint8)

        # Connected components cleanup
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_internal)
        min_area = 30
        clean_mask = np.zeros_like(mask_internal)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                clean_mask[labels==i] = 255
        mask_internal = clean_mask

        # ROI 업데이트
        roi_queue.clear()
        roi_queue.append({'edge': edges, 'internal': mask_internal})

        # 최신 processed_queue만
        processed_queue.clear()
        processed_queue.append((jpeg_bytes, depth_uint8))

# ------------------- Optical Flow Thread (최신 frame만) -------------------
def optical_flow_thread(flow_queue, thread_id):
    prev_gray = None
    prev_pts = None
    step = 3
    H,W = 128,128
    expansion = max(3, step)
    kernel = np.ones((expansion, expansion), dtype=np.uint8)

    while True:
        if not processed_queue:
            time.sleep(0.001)
            continue

        # 항상 최신 frame만
        jpeg_bytes, _ = processed_queue.pop()
        processed_queue.clear()

        cam_array = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        cam_frame = cv2.imdecode(cam_array, cv2.IMREAD_COLOR)
        if cam_frame is None:
            continue

        cam_resized = cv2.resize(cv2.UMat(cam_frame), (W,H))
        gray = cv2.cvtColor(cam_resized, cv2.COLOR_BGR2GRAY).get()

        if prev_gray is None:
            prev_gray = gray
            prev_pts = np.array([[x,y] for y in range(0,H,step) for x in range(0,W,step)],
                                dtype=np.float32).reshape(-1,1,2)
            flow_queue.clear()
            flow_queue.append({'mask': np.zeros((H,W),dtype=np.uint8),
                               'W_flow': np.zeros((H,W),dtype=np.float32)})
            continue

        # LK Optical Flow 계산
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None,
                                                       winSize=(10,10), maxLevel=2,
                                                       criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,10,0.03))
        if next_pts is None or status is None:
            prev_gray = gray.copy()
            continue

        dx = next_pts[:,0,0] - prev_pts[:,0,0]
        dy = next_pts[:,0,1] - prev_pts[:,0,1]
        flow_mag = np.sqrt(dx**2 + dy**2)
        valid = (status.flatten()==1)

        mask = np.zeros((H,W),dtype=np.uint8)
        W_flow = np.zeros((H,W),dtype=np.float32)
        if valid.any():
            ix = np.clip(prev_pts[valid,:,0].astype(int).flatten(),0,W-1)
            iy = np.clip(prev_pts[valid,:,1].astype(int).flatten(),0,H-1)
            mask[iy,ix] = 255
            max_flow = float(flow_mag[valid].max())
            if max_flow>1e-6:
                W_flow[iy,ix] = flow_mag[valid]/max_flow
            mask = cv2.dilate(cv2.UMat(mask), kernel,1).get()
            W_flow = cv2.dilate(cv2.UMat(W_flow.astype(np.float32)), kernel,1).get()
            W_flow = (W_flow/max(W_flow.max(),1e-6)).astype(np.float32)

        flow_queue.clear()
        flow_queue.append({'mask': mask, 'W_flow': W_flow, 'Max_flow':max_flow})

        prev_gray = gray.copy()
        prev_pts = next_pts[valid].reshape(-1,1,2) if valid.any() else None



# ==============================
# ------------------- 통합 스레드 개선 (Low/High Threshold + 화면 alignment) -------------------
def display_thread_with_threshold_aligned(target_H, target_W):
    MID_H, MID_W = 64, 64
    HALF_W = MID_W // 2   # 좌/우가 각각 32
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))

    theta = 0.5
    gamma = 1.0
    low_thr = 1.6

    while True:
        if not processed_queue:
            time.sleep(0.001)
            continue

        # --- 스마트폰에서 전송된 JPEG / depth ---
        jpeg_bytes, depth_uint8 = processed_queue.pop()

        # --- Optical Flow 입력 ---
        data1 = flow_queue_1[-1] if flow_queue_1 else {'W_flow': np.zeros((MID_H,HALF_W),np.float32), 
                                                       'mask': np.zeros((MID_H,HALF_W),np.uint8),
                                                       'Max_flow':1e-6}
        data2 = flow_queue_2[-1] if flow_queue_2 else {'W_flow': np.zeros((MID_H,HALF_W),np.float32), 
                                                       'mask': np.zeros((MID_H,HALF_W),np.uint8),
                                                       'Max_flow':1e-6}

        # --- resize ---
        W_flow_1 = cv2.resize(data1['W_flow'], (HALF_W, MID_H), interpolation=cv2.INTER_LINEAR)
        W_flow_2 = cv2.resize(data2['W_flow'], (HALF_W, MID_H), interpolation=cv2.INTER_LINEAR)
        mask_1 = cv2.resize(data1['mask'], (HALF_W, MID_H), interpolation=cv2.INTER_NEAREST)
        mask_2 = cv2.resize(data2['mask'], (HALF_W, MID_H), interpolation=cv2.INTER_NEAREST)

        max_flow_1 = data1.get('Max_flow', 1e-6)
        max_flow_2 = data2.get('Max_flow', 1e-6)

        # --- 지역 flow 복원 + 좌/우 합치기 ---
        flow_full = np.concatenate([W_flow_1 * max_flow_1, W_flow_2 * max_flow_2], axis=1)
        global_max = max(flow_full.max(), 1e-6)

        # --- mask 합치기 ---
        mask_full = np.concatenate([mask_1, mask_2], axis=1)

        # --- ROI ---
        roi = roi_queue[-1] if roi_queue else {
            'edge': np.zeros((MID_H,MID_W),np.float32),
            'internal': np.zeros((MID_H,MID_W),np.float32)
        }
        roi_edge = cv2.resize(roi['edge'], (MID_W, MID_H), interpolation=cv2.INTER_NEAREST)
        roi_internal = cv2.resize(roi['internal'], (MID_W, MID_H), interpolation=cv2.INTER_NEAREST)

        # --- depth & flow weight ---
        depth_mid = compute_depth_matrix(depth_uint8)
        k_depth = compute_depth_confidence(depth_uint8)

        # --- f_wflow 계산 (mask 기반 정적/동적 판단 포함) ---
        f_wflow = compute_flow_score_safe(flow_full/global_max, global_max, mask_full)

        W_flow_full = compute_flow_matrix(flow_full/global_max, f_wflow)
        k_flow = np.clip(compute_flow_weight(flow_full), 0.0, 1.0)

        # --- composite 계산 ---
        base = 0.63
        alpha = (6/1)*gamma/(base*0.7)
        beta = (3/1)*gamma/base

        composite = (
            alpha * k_depth * depth_mid +
            beta * k_flow * W_flow_full +
            gamma * roi_internal/255 +
            theta
        )

        mask_final = composite * roi_edge/255
        high_thr = 3.5

        # --- Low/High threshold mask ---
        mask_low = ((mask_final >= low_thr) & (mask_final < high_thr)).astype(np.float32)
        mask_high = (mask_final >= high_thr).astype(np.float32)

        kernel = np.ones((5, 5), np.uint8)
        mask_high_uint8 = mask_high.astype(np.uint8)
        mask_high_dilated = cv2.dilate(mask_high_uint8, kernel, iterations=1)

        # --- JPEG decode & 화면 맞춤 ---
        frame_arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        cam_frame = cv2.imdecode(frame_arr, cv2.IMREAD_COLOR)
        if cam_frame is None:
            continue

        Hf, Wf = cam_frame.shape[:2]
        cam_frame_resized = cv2.resize(cam_frame, (target_W, target_H), interpolation=cv2.INTER_LINEAR)
        mask_low_resized = cv2.resize(mask_low, (target_W, target_H), interpolation=cv2.INTER_NEAREST)
        mask_high_resized = cv2.resize(mask_high, (target_W, target_H), interpolation=cv2.INTER_NEAREST)
        mask_high_resized_dilated = cv2.resize(mask_high_dilated, (target_W, target_H), interpolation=cv2.INTER_NEAREST)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_high_resized_dilated, connectivity=8)

        MIN_AREA = int(target_W*target_H*0.03)
        boxes = []
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            if area >= MIN_AREA:  # 너무 작은건 제외
                 boxes.append((x, y, x+w, y+h))
        
        # --- Overlay ---
        overlay = cam_frame_resized.copy()
        mask_rgb = np.zeros_like(overlay, dtype=np.uint8)
        mask_rgb[mask_low_resized > 0] = [255,0,0]
        mask_rgb[mask_high_resized > 0] = [0,0,255]
        overlay = cv2.addWeighted(overlay, 0.5, mask_rgb, 0.5, 0)

        for (x1, y1, x2, y2) in boxes:
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 초록 박스

        cv2.imshow("ROI Display", overlay)
        if cv2.waitKey(1) & 0xFF in [ord('q'), ord('Q')]:
            break

    cv2.destroyAllWindows()

# ------------------- Main -------------------
def main():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST,PORT))
        s.listen(1)
        print(f"Listening on {HOST}:{PORT}")
        conn, addr = s.accept()
        print(f"Client connected: {addr}")

        # PC에서 보여줄 화면 크기
        target_H, target_W = 480, 640

        threading.Thread(target=receive_thread, args=(conn, raw_queue, gyro_queue, accel_queue), daemon=True).start()
        threading.Thread(target=depth_saliency_thread,daemon=True).start()
        threading.Thread(target=optical_flow_thread,args=(flow_queue_1,1),daemon=True).start()
        threading.Thread(target=optical_flow_thread,args=(flow_queue_2,2),daemon=True).start()
        t = threading.Thread(target=display_thread_with_threshold_aligned, args=(target_H, target_W))
        t.start()

        # keep main alive
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("Shutting down...")

if __name__=="__main__":
    main()
