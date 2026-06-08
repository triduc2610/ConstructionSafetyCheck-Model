import streamlit as st
import cv2
import math
import numpy as np
from PIL import Image
from ultralytics import YOLO
import tempfile
import time

st.set_page_config(
    page_title="Hệ thống Giám sát An toàn Công trường - Nhóm 5",
    page_icon="🏗️",
    layout="wide"
)

st.markdown("""
    <style>
        /* Căn giữa và làm nổi bật tiêu đề chính */
        h1 {
            color: #1e3a8a !important;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            font-weight: 700 !important;
            text-align: center;
            margin-bottom: 25px !important;
        }
        /* Làm đẹp cho tiêu đề ảnh cột con */
        h3 {
            font-weight: 600 !important;
        }
        /* Định dạng hộp chứa ảnh bo tròn góc và đổ bóng nhẹ (Giữ màu nền mặc định) */
        .img-box {
            padding: 10px;
            border-radius: 12px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
            margin-bottom: 15px;
        }
        /* Làm đẹp cấu trúc hiển thị số liệu KPIs bằng cách bo tròn góc */
        div[data-testid="stMetric"] {
            padding: 15px 20px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.04);
            border-left: 5px solid #3b82f6;
        }
        /* Tăng độ đậm cho chữ số KPI */
        div[data-testid="stMetricValue"] {
            font-weight: 700 !important;
        }
        /* Tạo khung lưới bọc ảnh bằng chứng vi phạm phía dưới */
        .evidence-card {
            border: 1px solid #ef4444;
            border-radius: 8px;
            padding: 5px;
            text-align: center;
            background-color: rgba(239, 68, 68, 0.02);
        }
    </style>
""", unsafe_allow_html=True)

# khởi tạo mô hình


@st.cache_resource
def load_custom_model(model_path):
    return YOLO(model_path)


# sidebar
st.sidebar.header("⚙️ Cấu hình Hệ thống")

selected_version = st.sidebar.selectbox(
    "🤖 Chọn phiên bản Mô hình AI",
    options=["YOLOv8 Nano (best.pt)", "YOLO11 Nano (best_yolo11.pt)"]
)

# Gán đường dẫn file trọng số tương ứng
model_file = "best.pt" if "YOLOv8" in selected_version else "best_yolo11.pt"
model = load_custom_model(model_file)
CLASS_NAMES = model.names

conf_threshold = st.sidebar.slider(
    "🎯 Độ tin cậy (Confidence)", 0.1, 1.0, 0.4, 0.05)
dist_threshold = st.sidebar.slider(
    "⚠️ Ngưỡng Vùng nguy hiểm (Pixels)", 50, 300, 150, 10)
uploaded_file = st.file_uploader(
    "📂 Chọn một hình ảnh công trường cần quét vi phạm...", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])

# bắt đầu xử lý
if uploaded_file is not None:
    # Phân tách luồng xử lý dựa trên định dạng tệp tin đầu vào
    file_type = uploaded_file.name.split('.')[-1].lower()

    # VIDEO
    if file_type in ["mp4", "avi", "mov"]:
        # Tạo file tạm thời lưu dữ liệu video thô để OpenCV có thể truy cập đọc luồng
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        # Khởi tạo đối tượng đọc video từ file tạm
        cap = cv2.VideoCapture(tfile.name)

        # Thiết lập vùng hiển thị giao diện động trên trang web Streamlit
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="img-box">', unsafe_allow_html=True)
            st.subheader("🎞️ Luồng Video đầu vào gốc")
            orig_placeholder = st.empty()
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="img-box">', unsafe_allow_html=True)
            st.subheader("🤖 Kết quả quét Real-time từ AI")
            res_placeholder = st.empty()
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("📊 Thống kê trạng thái tại khung hình")
        kpi_cols = st.columns(3)
        kpi1_placeholder = kpi_cols[0].empty()
        kpi2_placeholder = kpi_cols[1].empty()
        kpi3_placeholder = kpi_cols[2].empty()

        # BỘ NHỚ ẢNH BẰNG CHỨNG VI PHẠM
        st.markdown("---")
        st.markdown(
            "🚨 Danh sách ảnh bằng chứng vi phạm trích xuất")
        # Khung động hiển thị lưới ảnh bằng chứng phía dưới
        evidence_placeholder = st.empty()

        # Mảng chứa các bức ảnh vi phạm
        list_evidence_images = []
        # Mảng lưu tọa độ tâm của những người đã bị chụp ảnh
        captured_violation_centers = []

        # Từ điển theo dõi thời gian bắt đầu vi phạm của từng mục tiêu không gian
        # Cấu trúc lưu trữ: { (x_center, y_center): start_time_float }
        active_violation_trackers = {}

        # Ngưỡng thời gian duy trì lỗi liên tục (Xác nhận sau 1.5 giây vi phạm bền vững)
        CONFIRMATION_DELAY_SECONDS = 0.5

        # Vòng lặp phân tích từng khung hình của video
        while cap.isOpened():
            ret, frame_cv = cap.read()
            if not ret:
                break

            # Lưu lại khung hình gốc để hiển thị song song
            orig_frame = frame_cv.copy()
            orig_rgb = cv2.cvtColor(orig_frame, cv2.COLOR_BGR2RGB)
            orig_placeholder.image(orig_rgb, use_container_width=True)

            # tiến hành dự đoán trên khung hình hiện tại
            results = model(frame_cv, conf=conf_threshold, verbose=False)[0]

            # mảng lưu trữ đối tượng và bộ đếm
            workers = []
            machineries = []
            violation_summary = {"SAFE": 0,
                                 "PPE_VIOLATION": 0, "DANGER_ZONE": 0}

            # quét trích xuất và phân loại
            if results.boxes is not None:
                for box in results.boxes:
                    cls_id = int(box.cls[0])
                    label = CLASS_NAMES[cls_id]
                    coords = box.xyxy[0].tolist()
                    conf = float(box.conf[0])

                    # phân loại đối tượng Công nhân
                    if label == 'Person':
                        workers.append({
                            'box': coords,
                            'status': 'SAFE',
                            'color': (0, 255, 0),
                            'has_hardhat_error': False,
                            'has_vest_error': False,
                            'in_danger_zone': False
                        })

                    # phân loại phương tiện
                    elif label in ['Excavator', 'dump truck', 'machinery', 'vehicle', 'truck', 'wheel loader', 'trailer', 'sedan', 'van', 'SUV', 'bus']:
                        machineries.append({'box': coords, 'label': label})
                        x1, y1, x2, y2 = map(int, coords)
                        # khung của phương tiện
                        cv2.rectangle(frame_cv, (x1, y1),
                                      (x2, y2), (0, 165, 255), 2)
                        cv2.putText(frame_cv, f"{label} {conf:.2f}", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

                    # khung cho các thành phần an toàn
                    elif label in ['Hardhat', 'Safety Vest']:
                        x1, y1, x2, y2 = map(int, coords)
                        cv2.rectangle(frame_cv, (x1, y1),
                                      (x2, y2), (0, 255, 0), 1)

                # phân tích không gian an toàn
                # quét lỗi vi phạm PPE
                for box in results.boxes:
                    lbl = CLASS_NAMES[int(box.cls[0])]
                    if lbl in ['NO-Hardhat', 'NO-Safety Vest']:
                        cx1, cy1, cx2, cy2 = box.xyxy[0].tolist()
                        c_center_x = (cx1 + cx2) / 2
                        c_center_y = (cy1 + cy2) / 2

                        # kiểm tra tâm lỗi vi phạm lọt vào trong khung bao của người nào
                        for worker in workers:
                            wx1, wy1, wx2, wy2 = worker['box']
                            if wx1 <= c_center_x <= wx2 and wy1 <= c_center_y <= wy2:
                                if lbl == 'NO-Hardhat':
                                    worker['has_hardhat_error'] = True
                                if lbl == 'NO-Safety Vest':
                                    worker['has_vest_error'] = True
                                worker['color'] = (0, 0, 255)

                # tính khoảng cách nguy hiểm giữa người và phương tiện
                danger_violations = []
                for worker in workers:
                    wx1, wy1, wx2, wy2 = worker['box']
                    w_centroid = (int((wx1 + wx2) / 2), int((wy1 + wy2) / 2))

                    for machine in machineries:
                        mx1, my1, mx2, my2 = machine['box']
                        m_centroid = (int((mx1 + mx2) / 2),
                                      int((my1 + my2) / 2))

                        distance = math.sqrt(
                            (w_centroid[0] - m_centroid[0])**2 + (w_centroid[1] - m_centroid[1])**2)

                        if distance < dist_threshold:
                            worker['in_danger_zone'] = True
                            worker['color'] = (0, 0, 255)
                            danger_violations.append((w_centroid, m_centroid))

                # tổng hợp kết quả
                for worker in workers:
                    # mảng vi phạm
                    errors = []
                    if worker['has_hardhat_error']:
                        errors.append("NO-Hardhat")
                    if worker['has_vest_error']:
                        errors.append("NO-Vest")

                    if worker['in_danger_zone']:
                        if errors:
                            worker['status'] = f"PPE ERROR ({', '.join(errors)}) + DANGER"
                        else:
                            worker['status'] = "DANGER ZONE"
                    else:
                        if errors:
                            worker['status'] = f"PPE VIOLATION ({', '.join(errors)})"
                        else:
                            worker['status'] = "SAFE"

                    # vẽ khung công nhân
                    x1, y1, x2, y2 = map(int, worker['box'])
                    cv2.rectangle(frame_cv, (x1, y1),
                                  (x2, y2), worker['color'], 3)
                    cv2.putText(frame_cv, worker['status'], (x1, y1 - 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, worker['color'], 2)

                    # phân chia thống kê về bộ đếm
                    if worker['in_danger_zone'] and (worker['has_hardhat_error'] or worker['has_vest_error']):
                        violation_summary["PPE_VIOLATION"] += 1
                        violation_summary["DANGER_ZONE"] += 1
                    elif worker['has_hardhat_error'] or worker['has_vest_error']:
                        violation_summary["PPE_VIOLATION"] += 1
                    elif worker['in_danger_zone']:
                        violation_summary["DANGER_ZONE"] += 1
                    else:
                        violation_summary["SAFE"] += 1

                # vẽ đường nối khoảng cách nguy hiểm giữa người và máy móc
                for w_c, m_c in danger_violations:
                    cv2.line(frame_cv, w_c, m_c, (0, 0, 255), 3)

            # chuyển đổi về hệ màu RGB và render kết quả động ra màn hình web
            result_image = cv2.cvtColor(frame_cv, cv2.COLOR_BGR2RGB)
            res_placeholder.image(result_image, use_container_width=True)

            # TỰ ĐỘNG CHỤP ẢNH BẰNG CHỨNG
            current_loop_violation_centers = []
            current_timestamp = time.time()

            for worker in workers:
                if worker['status'] != "SAFE":
                    wx1, wy1, wx2, wy2 = worker['box']
                    w_center = (int((wx1 + wx2) / 2), int((wy1 + wy2) / 2))
                    current_loop_violation_centers.append(w_center)

                    matched_tracker_center = None
                    for tracked_center in list(active_violation_trackers.keys()):
                        spatial_dist = math.sqrt(
                            (w_center[0] - tracked_center[0])**2 + (w_center[1] - tracked_center[1])**2)
                        if spatial_dist < 40:
                            matched_tracker_center = tracked_center
                            break

                    if matched_tracker_center is None:
                        active_violation_trackers[w_center] = current_timestamp
                        violation_duration = 0.0
                        tracker_key = w_center
                    else:
                        violation_duration = current_timestamp - \
                            active_violation_trackers[matched_tracker_center]
                        tracker_key = matched_tracker_center

                    if violation_duration >= CONFIRMATION_DELAY_SECONDS:
                        is_already_captured = False
                        for past_center in captured_violation_centers:
                            spatial_dist = math.sqrt(
                                (w_center[0] - past_center[0])**2 + (w_center[1] - past_center[1])**2)
                            if spatial_dist < 40:
                                is_already_captured = True
                                break

                        if not is_already_captured:
                            captured_violation_centers.append(w_center)
                            timestamp_str = time.strftime(
                                '%H:%M:%S', time.localtime())

                            list_evidence_images.insert(
                                0, {"img": result_image.copy(), "time": timestamp_str, "type": worker['status']})
                            if len(list_evidence_images) > 8:
                                list_evidence_images.pop()

            for tracked_center in list(active_violation_trackers.keys()):
                still_active = False
                for current_center in current_loop_violation_centers:
                    spatial_dist = math.sqrt(
                        (current_center[0] - tracked_center[0])**2 + (tracked_center[1] - current_center[1])**2)
                    if spatial_dist < 40:
                        still_active = True
                        break
                if not still_active:
                    del active_violation_trackers[tracked_center]

            if list_evidence_images:
                with evidence_placeholder.container():
                    cols_ev = st.columns(4)
                    for idx, ev_data in enumerate(list_evidence_images):
                        col_idx = idx % 4
                        with cols_ev[col_idx]:
                            st.markdown(f'<div class="evidence-card">',
                                        unsafe_allow_html=True)
                            st.image(ev_data["img"], use_container_width=True)
                            st.caption(
                                f"🚨 {ev_data['type']} | ⏰ {ev_data['time']}")
                            st.markdown('</div>', unsafe_allow_html=True)

            kpi1_placeholder.metric(
                label="✅ Công nhân AN TOÀN", value=violation_summary["SAFE"])
            kpi2_placeholder.metric(
                label="🚨 Vi phạm bảo hộ (PPE)", value=violation_summary["PPE_VIOLATION"])
            kpi3_placeholder.metric(
                label="❌ Vi phạm Vùng nguy hiểm", value=violation_summary["DANGER_ZONE"])

        cap.release()

    # HÌNH ẢNH
    else:
        image = Image.open(uploaded_file)
        frame = np.array(image)  # chuyển thành mảng NumPy

        # đổi từ RGB (PIL) sang BGR (OpenCV)
        frame_cv = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # tiến hành dự đoán
        results = model(frame_cv, conf=conf_threshold)[0]

        # mảng lưu trữ đối tượng và bộ đếm
        workers = []
        machineries = []
        violation_summary = {"SAFE": 0, "PPE_VIOLATION": 0, "DANGER_ZONE": 0}

        # quét trích xuất và phân loại
        if results.boxes is not None:
            for box in results.boxes:
                cls_id = int(box.cls[0])
                label = CLASS_NAMES[cls_id]
                coords = box.xyxy[0].tolist()
                conf = float(box.conf[0])

                # phân loại đối tượng Công nhân
                if label == 'Person':
                    workers.append({
                        'box': coords,
                        'status': 'SAFE',
                        'color': (0, 255, 0),
                        'has_hardhat_error': False,
                        'has_vest_error': False,
                        'in_danger_zone': False
                    })

                # phân loại phương tiện
                elif label in ['Excavator', 'dump truck', 'machinery', 'vehicle', 'truck', 'wheel loader', 'trailer', 'sedan', 'van', 'SUV', 'bus']:
                    machineries.append({'box': coords, 'label': label})
                    x1, y1, x2, y2 = map(int, coords)
                    # khung của phương tiện
                    cv2.rectangle(frame_cv, (x1, y1),
                                  (x2, y2), (0, 165, 255), 2)
                    cv2.putText(frame_cv, f"{label} {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

                # khung cho các thành phần an toàn
                elif label in ['Hardhat', 'Safety Vest']:
                    x1, y1, x2, y2 = map(int, coords)
                    cv2.rectangle(frame_cv, (x1, y1), (x2, y2), (0, 255, 0), 1)

            # phân tích không gian an toàn
            # quét lỗi vi phạm PPE
            for box in results.boxes:
                lbl = CLASS_NAMES[int(box.cls[0])]
                if lbl in ['NO-Hardhat', 'NO-Safety Vest']:
                    cx1, cy1, cx2, cy2 = box.xyxy[0].tolist()
                    c_center_x = (cx1 + cx2) / 2
                    c_center_y = (cy1 + cy2) / 2

                    # kiểm tra tâm lỗi vi phạm lọt vào trong khung bao của người nào
                    for worker in workers:
                        wx1, wy1, wx2, wy2 = worker['box']
                        if wx1 <= c_center_x <= wx2 and wy1 <= c_center_y <= wy2:
                            if lbl == 'NO-Hardhat':
                                worker['has_hardhat_error'] = True
                            if lbl == 'NO-Safety Vest':
                                worker['has_vest_error'] = True
                            worker['color'] = (0, 0, 255)

            # tính khoảng cách nguy hiểm giữa người và phương tiện
            danger_violations = []
            for worker in workers:
                wx1, wy1, wx2, wy2 = worker['box']
                w_centroid = (int((wx1 + wx2) / 2), int((wy1 + wy2) / 2))

                for machine in machineries:
                    mx1, my1, mx2, my2 = machine['box']
                    m_centroid = (int((mx1 + mx2) / 2),
                                  int((my1 + my2) / 2))

                    distance = math.sqrt(
                        (w_centroid[0] - m_centroid[0])**2 + (w_centroid[1] - m_centroid[1])**2)

                    if distance < dist_threshold:
                        worker['in_danger_zone'] = True
                        worker['color'] = (0, 0, 255)
                        danger_violations.append((w_centroid, m_centroid))

            # tổng hợp kết quả
            for worker in workers:
                # mảng vi phạm
                errors = []
                if worker['has_hardhat_error']:
                    errors.append("NO-Hardhat")
                if worker['has_vest_error']:
                    errors.append("NO-Vest")

                if worker['in_danger_zone']:
                    if errors:
                        worker['status'] = f"PPE ERROR ({', '.join(errors)}) + DANGER"
                    else:
                        worker['status'] = "DANGER ZONE"
                else:
                    if errors:
                        worker['status'] = f"PPE VIOLATION ({', '.join(errors)})"
                    else:
                        worker['status'] = "SAFE"

                # vẽ khung công nhân
                x1, y1, x2, y2 = map(int, worker['box'])
                cv2.rectangle(frame_cv, (x1, y1), (x2, y2), worker['color'], 3)
                cv2.putText(frame_cv, worker['status'], (x1, y1 - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, worker['color'], 2)

                # phân chia thống kê về bộ đếm
                if worker['in_danger_zone'] and (worker['has_hardhat_error'] or worker['has_vest_error']):
                    violation_summary["PPE_VIOLATION"] += 1
                    violation_summary["DANGER_ZONE"] += 1
                elif worker['has_hardhat_error'] or worker['has_vest_error']:
                    violation_summary["PPE_VIOLATION"] += 1
                elif worker['in_danger_zone']:
                    violation_summary["DANGER_ZONE"] += 1
                else:
                    violation_summary["SAFE"] += 1

            # vẽ đường nối khoảng cách nguy hiểm giữa người và máy móc
            for w_c, m_c in danger_violations:
                cv2.line(frame_cv, w_c, m_c, (0, 0, 255), 3)

        # chuyển đổi về hệ màu RGB
        result_image = cv2.cvtColor(frame_cv, cv2.COLOR_BGR2RGB)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="img-box">', unsafe_allow_html=True)
            st.subheader("📸 Ảnh gốc đầu vào")
            st.image(image, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="img-box">', unsafe_allow_html=True)
            st.subheader("🤖 Kết quả phân tích từ AI")
            st.image(result_image, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("### 📊 Thống kê trạng thái tại khung hình")
        kpi1, kpi2, kpi3 = st.columns(3)
        with kpi1:
            st.metric(label="✅ Công nhân AN TOÀN",
                      value=violation_summary["SAFE"])
        with kpi2:
            st.metric(label="🚨 Vi phạm bảo hộ (PPE)",
                      value=violation_summary["PPE_VIOLATION"])
        with kpi3:
            st.metric(label="❌ Vi phạm Vùng nguy hiểm",
                      value=violation_summary["DANGER_ZONE"])
