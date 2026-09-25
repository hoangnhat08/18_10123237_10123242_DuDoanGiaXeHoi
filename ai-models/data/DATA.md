# Nguồn dữ liệu

- Nguồn: Kaggle – `danh911/gi-xe` (crawl từ bonbanh.com).
- Giấy phép: theo giấy phép công khai của bộ dữ liệu trên Kaggle (xem trang dataset để biết chi tiết/điều khoản sử dụng).
- Số dòng gốc: ~26.956. Sau làm sạch (`ai-models/src/data_utils.py`): 24.723 dòng.
- Đơn vị giá (`price`) sau khi parse: **triệu VNĐ**.
- Cách giải nén: `unzip dataset.zip -d .` sẽ tạo ra `data.csv` cùng thư mục.

## Mô tả cột

| Cột | Ý nghĩa |
|---|---|
| car_name | Tên đầy đủ tin đăng (dùng để tách brand/model) |
| year | Năm sản xuất |
| price | Giá rao bán (chuỗi, ví dụ "2 Tỷ 700 Triệu") |
| assemble_place | Nơi lắp ráp (trong nước / nhập khẩu) |
| series | Kiểu dáng (Sedan, SUV, ...) |
| driven kms | Số km đã đi |
| num_of_door | Số cửa |
| num_of_seat | Số chỗ ngồi |
| engine_type | Loại nhiên liệu |
| transmission | Loại hộp số |
| url | Link tin đăng gốc (dùng làm khoá loại trùng) |
