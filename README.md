# 07_MSSV1_MSSV2_HocMayCoBan — Dự đoán giá xe ô tô cũ

> ⚠️ Thay `07_MSSV1_MSSV2_HocMayCoBan` bằng đúng tên nhóm/MSSV của bạn theo quy ước lớp, và điền các mục còn để trống (thành viên, địa chỉ public, v.v.) trước khi nộp.

## 1. Thành viên

| Họ tên | MSSV | Phần việc |
|---|---|---|
| *(điền)* | *(điền)* | Training / EDA / báo cáo |
| *(điền)* | *(điền)* | AI Service / Backend / Frontend / Docker |

## 2. Bài toán

- **Loại bài toán:** Hồi quy (Regression).
- **Cột mục tiêu:** `price` — giá rao bán xe cũ, đơn vị **triệu VNĐ**.
- **Ý nghĩa thực tế:** ước lượng nhanh mức giá hợp lý cho một chiếc xe cũ dựa trên đặc điểm của nó (hãng, dòng xe, năm sản xuất, số km đã đi, xuất xứ, nhiên liệu, hộp số, số cửa, số chỗ), hỗ trợ người mua/bán tham khảo giá thị trường.

## 3. Dữ liệu

- Nguồn: Kaggle `danh911/gi-xe` (crawl từ bonbanh.com). Chi tiết giấy phép và mô tả cột: [`ai-models/data/DATA.md`](ai-models/data/DATA.md).
- Giải nén: `unzip ai-models/data/dataset.zip -d ai-models/data/` → tạo ra `data.csv`.
- ~27.000 dòng gốc, còn **24.723 dòng** sau làm sạch (xem `ai-models/src/data_utils.py`).

## 4. Kết quả model

**Cả 4 model được giữ riêng biệt** (không chỉ chọn 1 model duy nhất) — mỗi lần dự đoán, giao diện hiển thị đủ kết quả của cả 4 model để so sánh trực tiếp, model tốt nhất được đánh dấu.

| Model | MAE (triệu) | RMSE (triệu) | R² (test) | File |
|---|---:|---:|---:|---|
| **SVR (tốt nhất)** | **83.3** | **289.0** | **0.9772** | `model_svr.joblib` |
| Linear Regression (baseline) | 100.2 | 344.7 | 0.9676 | `model_linear_regression.joblib` |
| Random Forest | 136.1 | 457.5 | 0.9429 | `model_random_forest.joblib` |
| Decision Tree | 217.1 | 611.6 | 0.8980 | `model_decision_tree.joblib` |

**SVR** được đánh dấu tốt nhất (`is_best`) vì MAE/RMSE thấp nhất, R² cao nhất, kích thước file nhỏ (0.18 MB nén) — phù hợp deploy trên máy cá nhân/Colab/tunnel có RAM hạn chế. Nhưng khi dự đoán, **AI Service vẫn chạy cả 4 model và trả về đủ 4 kết quả**, không chỉ trả kết quả của model tốt nhất. Chi tiết lý do và biểu đồ phân dư: [`ai-models/colab/04_evaluate.ipynb`](ai-models/colab/04_evaluate.ipynb), bảng đầy đủ: [`docs/metrics.csv`](docs/metrics.csv).

## 5. Đóng gói model

- **Mỗi model là một file `.joblib` riêng** trong `ai-models/models/`: `model_linear_regression.joblib`, `model_decision_tree.joblib`, `model_random_forest.joblib`, `model_svr.joblib`. Mỗi file chứa cả pipeline tiền xử lý (`ColumnTransformer`: `StandardScaler` cho biến số, `OneHotEncoder` cho biến phân loại) **và** model tương ứng, bọc trong `TransformedTargetRegressor` (log1p/expm1 cho `price`) — tiền xử lý lúc dự đoán giống hệt lúc huấn luyện.
- `model.joblib` được giữ lại như **bản sao (alias)** của model tốt nhất (SVR) để tiện xem nhanh; AI Service **không** dùng riêng file này để trả kết quả mà nạp đủ cả 4 file `model_<key>.joblib` ở trên.
- `ai-models/models/metadata.json` có mảng `"models"`: liệt kê `key`, `name`, `file`, `is_best`, metric của từng model — đây là **nguồn sự thật duy nhất** để AI Service biết cần nạp file nào ứng với model nào; thêm/bớt model chỉ cần sửa `train.py`/`metadata.json`, không cần sửa code AI Service.
- Kèm `ai-models/models/schema.json` (đặc trưng, kiểu, khoảng giá trị, nhãn hợp lệ — dùng chung cho cả 4 model vì cùng pipeline đặc trưng).
- **Cách export từ Colab:** chạy `ai-models/colab/03_train.ipynb` trên Colab (upload/giải nén `dataset.zip` trước) → notebook tự lưu đủ 4 file `model_<key>.joblib` + `model.joblib` (alias) → tải về (`files.download(...)`) hoặc lưu Google Drive → copy cả 4+1 file vào đúng đường dẫn `ai-models/models/` trong repo → `git add ai-models/models/ && git commit && git push`. AI Service sẽ nạp đúng các file này khi container khởi động.

## 6. Kiến trúc hệ thống

```
Người dùng → Frontend (nginx :3000) → Backend (Flask :8000) → AI Service (Flask :8001)
                                                                  nạp cả 4 model_<key>.joblib
                                                                  /predict trả về kết quả của cả 4
```

- **AI Service** (`ai-models/service/app.py`): nạp **cả 4 file model** ngay khi container khởi động (không đợi request đầu tiên); `POST /predict` trả về mảng `results` gồm kết quả của cả 4 model kèm `best_model`; `GET /health`, `GET /model-info`, `GET /options`.
- **Backend** (`app/backend/app.py`): validate theo `schema.json` (lấy từ AI Service), gọi AI Service, **chuyển tiếp nguyên kết quả của cả 4 model** cho Frontend, lưu lịch sử; `POST /api/predict`, `GET /api/history`, `GET /health`.
- **Frontend** (`app/frontend/index.html`): form nhập liệu tĩnh, gọi `/api/predict` qua nginx reverse-proxy, **hiển thị 4 thẻ kết quả** (mỗi model 1 thẻ, model tốt nhất được đánh dấu).

## 7. Chạy trên máy

Yêu cầu: đã cài Docker + Docker Compose.

```bash
cp .env.example .env
docker compose up --build
```

- Frontend: http://localhost:3000
- Backend: http://localhost:8000/health
- AI Service (nội bộ, hoặc mở port để test riêng): http://localhost:8001/health

Dừng: `docker compose down`.

**Chạy không dùng Docker** (debug từng service):

```bash
# terminal 1
cd ai-models && pip install -r requirements.txt && AI_SERVICE_PORT=8001 python service/app.py
# terminal 2
cd app/backend && pip install -r requirements.txt && AI_SERVICE_URL=http://localhost:8001 SERVER_PORT=8000 python app.py
# mo app/frontend/index.html truc tiep, hoac serve bang: python -m http.server 3000
```

## 8. Huấn luyện lại model

- Notebook Colab, chạy theo thứ tự trong `ai-models/colab/`: `01_eda.ipynb` → `02_preprocess.ipynb` → `03_train.ipynb` → `04_evaluate.ipynb`.
- Hoặc chạy script tương đương (không cần Colab):

```bash
cd ai-models/src
python eda.py     # sinh hinh EDA -> docs/figures/
python train.py   # huan luyen ca 4 model, xuat 4 file model_<key>.joblib rieng
                   # (+ model.joblib la alias cua model tot nhat) + schema.json + metadata.json
```

> ⚠️ Ghim đúng phiên bản `scikit-learn`/`numpy`/`pandas` như `ai-models/requirements.txt` để cả 4 file `model_*.joblib` load được trong Docker.

### Ví dụ gọi API — nhận đủ kết quả của cả 4 model

```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {"brand":"Toyota","model":"Camry 2.5Q","series":"Sedan",
       "year":2018,"driven_kms":50000,"assemble_place":"Nhập khẩu",
       "engine_type":"Xăng","transmission":"Số tự động",
       "num_of_door":4,"num_of_seat":5}}'
```

```jsonc
{
  "best_model": { "key": "svr", "model_name": "SVR", "is_best": true,
                  "prediction_million_vnd": 872.2, "prediction_text": "872 triệu VNĐ" },
  "results": [
    { "key": "svr", "model_name": "SVR", "is_best": true,
      "prediction_million_vnd": 872.2, "prediction_text": "872 triệu VNĐ",
      "predict_time_ms": 5.4, "metrics_test": {"MAE": 83.3, "RMSE": 289.05, "R2": 0.9772} },
    { "key": "linear_regression", "model_name": "Linear Regression (baseline)", "is_best": false,
      "prediction_million_vnd": 898.7, "prediction_text": "899 triệu VNĐ", "predict_time_ms": 6.6 },
    { "key": "decision_tree", "model_name": "Decision Tree", "is_best": false,
      "prediction_million_vnd": 729.0, "prediction_text": "729 triệu VNĐ", "predict_time_ms": 4.6 },
    { "key": "random_forest", "model_name": "Random Forest", "is_best": false,
      "prediction_million_vnd": 747.0, "prediction_text": "747 triệu VNĐ", "predict_time_ms": 18.0 }
  ],
  "model_version": "2.0.0",
  "request_id": "358152cd"
}
```

Giao diện web (`app/frontend/index.html`) hiển thị `results` dưới dạng **4 thẻ kết quả**, thẻ của model trong `best_model` được viền nổi bật + gắn nhãn "Tốt nhất".

## 9. Biến môi trường

| Biến | Ý nghĩa |
|---|---|
| `SERVER_PORT` | Cổng Backend chạy trong container |
| `AI_SERVICE_PORT` | Cổng AI Service chạy trong container |
| `AI_SERVICE_URL` | Địa chỉ Backend gọi tới AI Service — luôn là tên service nội bộ `http://ai-service:8001`, **không đổi** khi public bằng ngrok (Backend không lộ ra ngoài trực tiếp) |
| `PUBLIC_APP_URL` | Link ngrok/tunnel hiện tại trỏ tới Frontend (cổng 3000) — do `scripts/start_ngrok.sh` tự ghi mỗi lần chạy; dùng để dán vào mục 11 README, không được code đọc trực tiếp |
| `PUBLIC_AI_SERVICE_URL` | (tuỳ chọn) link tunnel riêng cho AI Service nếu muốn test độc lập `/model-info`, `/docs` từ bên ngoài |
| `MONGODB_URI` | (tuỳ chọn) chuỗi kết nối MongoDB nếu thay lịch sử trong bộ nhớ bằng lưu trữ lâu dài |

> Ghi chú: Frontend (nginx) không đọc `.env` để lấy địa chỉ Backend — nó dùng tên service Docker `http://backend:8000` cố định trong `Dockerfile`/`nginx.conf.template` (đúng khuyến nghị "dùng tên service trong Docker network, không hardcode localhost/IP **public**"). Khi public bằng ngrok, chỉ địa chỉ **bên ngoài** (`PUBLIC_APP_URL`) thay đổi — mọi kết nối **bên trong** docker-compose network giữ nguyên tên service.

## 10. Triển khai (public bằng ngrok)

Nhóm chọn cách **chạy Docker trên máy cá nhân + public bằng ngrok** (không có IP tĩnh).

**Vì sao chỉ cần public 1 link duy nhất (Frontend):** Frontend (nginx) đã tự proxy `/api/*` sang Backend qua tên service `backend` trong mạng Docker nội bộ, Backend lại gọi AI Service qua tên service `ai-service` — không service nào hardcode `localhost`/IP. Vì vậy chỉ cần mở tunnel cho **cổng 3000 (Frontend)** là đủ để người ngoài dùng trọn luồng FE → BE → AI Service.

**Các bước:**

```bash
# 1) Cai dat (lam 1 lan)
cp .env.example .env
# cai ngrok: https://ngrok.com/download, sau do dang nhap:
ngrok config add-authtoken <TOKEN_CUA_BAN>

# 2) Bat he thong
docker compose up -d --build
docker compose ps                 # kiem tra ca 3 service da "healthy"

# 3) Bat tunnel (tu dong ghi log + cap nhat .env moi lan link doi)
./scripts/start_ngrok.sh          # mac dinh public cong 3000
```

Script `scripts/start_ngrok.sh` sẽ:
- Lấy link `https://xxxx.ngrok-free.app` mới nhất từ ngrok.
- **Tự ghi log** thời điểm đổi + địa chỉ cũ/mới vào `logs/tunnel-changes.log` (đáp ứng yêu cầu bắt buộc "mỗi lần cổng/tunnel đổi phải ghi log").
- Tự cập nhật biến `PUBLIC_APP_URL` trong `.env`.
- In ra nhắc nhở 2 việc thủ công còn lại: cập nhật mục 11–12 bên dưới trong README, và `git commit && git push`.

**Kiểm tra trước khi báo là xong:** mở link ngrok bằng cửa sổ ẩn danh, nhập 1 mẫu dữ liệu, bấm dự đoán, xem cả 4 kết quả model trả về đúng — tức là luồng Frontend → Backend → AI Service đã thông trên địa chỉ public thật.

**Vì ngrok bản miễn phí không có IP tĩnh** (link đổi mỗi lần chạy lại): phải chạy lại `docker compose up -d` + `./scripts/start_ngrok.sh` và cập nhật link mới **vào mỗi sáng thứ Hai** trước khi giảng viên kiểm tra định kỳ (xem `logs/tunnel-changes.log` để biết lần cập nhật gần nhất).

*(Phương án thay thế nếu sau này có tài khoản ngrok trả phí hoặc muốn deploy cloud: dùng domain cố định của ngrok, hoặc deploy Backend/AI Service lên Render + Frontend lên Vercel — không bắt buộc cho đợt này.)*

## 11. Demo online

- App (Frontend, dùng luôn được cả hệ thống): *(dán link ngrok mới nhất từ `PUBLIC_APP_URL` trong `.env`, ví dụ `https://xxxx.ngrok-free.app`)*
- AI Service (không bắt buộc public riêng — có thể xem qua Máy 2/TeamViewer khi bảo vệ): `http://ai-service:8001/health`, `/model-info` (nội bộ trong docker network)

## 12. Nhật ký đổi cổng/tunnel

> Bảng này nên đồng bộ với file `logs/tunnel-changes.log` (được `scripts/start_ngrok.sh` tự ghi mỗi lần chạy). Copy dòng mới nhất từ file log đó vào đây rồi commit.

| Thời điểm | Địa chỉ cũ | Địa chỉ mới |
|---|---|---|
| *(điền, xem `logs/tunnel-changes.log`)* | | |

## 13. Kết quả kiểm thử hiệu năng

*(điền sau khi chạy load test — xem gợi ý công cụ k6/Locust/Apache Bench trong tài liệu môn học)*

## 14. Hạn chế và hướng phát triển

- Dữ liệu là **giá rao bán**, không phải giá giao dịch thực tế.
- Model gặp hãng/dòng xe chưa từng thấy vẫn dự đoán được nhờ `handle_unknown="ignore"`, nhưng độ chính xác giảm.
- Lịch sử dự đoán hiện lưu trong bộ nhớ (mất khi restart Backend) — có thể thay bằng MongoDB khi cần lưu lâu dài/chia sẻ nhiều instance.
- Hướng phát triển: thêm xác thực người dùng, cache kết quả theo đặc trưng, `GridSearchCV` mở rộng hơn cho SVR/Random Forest, thêm ảnh xe vào dự đoán (multi-modal).
