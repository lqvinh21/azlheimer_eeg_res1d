# Alzheimer EEG - CAUEEG, 1D-ResNet (AD vs HC)

Project Python chạy trực tiếp trên workstation Linux/Windows, chuyển từ notebook Kaggle `eegadhcmci` (v3). Không còn magic notebook (`!pip`), không còn đường dẫn `/kaggle/...`; mọi đường dẫn và siêu tham số đi qua dòng lệnh. Mỗi file phụ trách một phần của quy trình.

Bài toán: phân loại **Normal (HC, nhãn 0)** và **Dementia/AD (nhãn 1)** từ EEG 19 kênh của dataset CAUEEG (Kim, Youn, Paik - NeuroImage 2023). Bỏ lớp MCI.

## Cấu trúc thư mục

```text
alzheimer_eeg_caueeg/
├── README.md                         # Cài đặt, bố cục dữ liệu, cách chạy
├── requirements.txt                  # Thư viện Python (torch cài riêng)
├── train.py                          # CLI chính; nối các khối của project
└── eeg_caueeg/
    ├── __init__.py
    ├── config.py                     # Kênh EEG, hằng số tuổi, TrainConfig (siêu tham số)
    ├── data/
    │   ├── __init__.py
    │   ├── split.py                  # Đọc dementia-no-overlap.json, nhãn HC/AD, bỏ MCI
    │   ├── edf.py                    # Đọc EDF: chọn 19 kênh, band-pass 0.5-45 Hz, resample, cache .npy
    │   ├── records.py                # RecordStore: tín hiệu, nhãn, tuổi, symptom theo serial
    │   ├── windows.py                # Z-score theo fold, Dataset cửa sổ train/eval, DataLoader
    │   └── folds.py                  # StratifiedGroupKFold trên train+val (DEV)
    ├── models/
    │   ├── __init__.py
    │   └── resnet1d.py               # Pre-act 1D-ResNet (GroupNorm) + nhánh tuổi, EMA
    ├── engine/
    │   ├── __init__.py
    │   ├── inference.py              # Logit mức record, AUC, loss/acc/AUC
    │   ├── train_fold.py             # Huấn luyện một fold, ghi lịch sử train/val theo epoch
    │   ├── cv.py                     # Vòng CV theo biến thể, lưu/tiếp tục từng fold (.pkl)
    │   ├── age_baseline.py           # Baseline chỉ dùng tuổi (cùng fold)
    │   └── reporting.py              # OOF/test, bootstrap CI, Youden, phân tích lỗi, classification report
    └── utils/
        ├── __init__.py
        ├── seed.py                   # Cố định seed
        └── plots.py                  # ROC, đường cong train-val (acc/AUC/loss), confusion matrix
```

## Quy trình

1. `data/split.py` đọc split chính thức `no-overlap` (train 367 HC / 249 AD; test đã loại bệnh nhân trùng với train).
2. `data/edf.py` + `data/records.py` nạp EDF, lọc 0.5-45 Hz, resample 128 Hz, cache `.npy`. Z-score **không** tính ở đây.
3. `data/folds.py` chia `StratifiedGroupKFold` trên train+val (DEV, 670 record). Fold cố định, dùng chung cho mọi biến thể.
4. `engine/train_fold.py` huấn luyện số epoch cố định, lấy trọng số EMA cuối (không early stopping, không chọn checkpoint theo val). Z-score tính riêng từ phần train của fold. Mỗi record được gộp bằng trung bình logit (AD - HC) trên 24 cửa sổ đều nhau.
5. `engine/cv.py` chạy hai biến thể: `age` (EEG + tuổi) và `no_age` (chỉ EEG). Chạy lại sẽ tiếp tục từ fold còn thiếu.
6. `engine/age_baseline.py` baseline chỉ dùng tuổi để biết EEG đóng góp bao nhiêu ngoài tuổi.
7. `engine/reporting.py` báo cáo **OOF** (mỗi record DEV được dự đoán bởi model không thấy nó) và **test chính thức** (ensemble 5 model fold). AUC kèm bootstrap 95% CI; ngưỡng Youden chọn trên OOF, không chọn trên test.
8. `utils/plots.py` lưu đồ thị: ROC, AUC held-out theo epoch, train-vs-val acc/AUC/loss cho từng biến thể, confusion matrix OOF.

## Cài đặt

Cài PyTorch (kèm CUDA phù hợp) theo https://pytorch.org/get-started/locally/ rồi:

```bash
python -m pip install -r requirements.txt
```

## Tải dữ liệu và bài báo CAUEEG

Truy cập [thư mục Google Drive CAUEEG](https://drive.google.com/drive/folders/1JDNKGCZFsWef2EUwZw2KRzda8j1PrtvT) để tải bộ dữ liệu và bài báo về bộ dữ liệu. Nếu Drive yêu cầu quyền truy cập, hãy đăng nhập tài khoản được cấp quyền.

Sau khi tải, giải nén dữ liệu và kiểm tra thư mục chứa file `dementia-no-overlap.json` cùng thư mục `signal/edf/`. Dùng đường dẫn tới thư mục đó làm `--data-root` khi chạy chương trình. Bài báo được cung cấp để đọc và tham khảo; chương trình không cần file bài báo để chạy.

Khi chạy, chương trình đọc thông tin các bản ghi và split từ `dementia-no-overlap.json`, rồi nạp các file EEG `.edf` tương ứng trong `signal/edf/`. Tín hiệu được chọn 19 kênh, lọc 0.5-45 Hz, resample về 128 Hz và lưu cache `.npy` để dùng lại ở những lần chạy sau.

## Bố cục dữ liệu

`--data-root` trỏ tới thư mục CAUEEG có:

```text
CauEEG_AD+HC+MCI_EEG/
├── dementia-no-overlap.json
└── signal/
    └── edf/
        ├── 00001.edf
        └── ...
```

Có thể chỉ định riêng bằng `--split-file` và `--edf-root`.

## Chạy

```bash
python train.py --data-root "/path/to/CauEEG_AD+HC+MCI_EEG" --output-dir "./caueeg_output"
```

Tuỳ chọn thường dùng:

```bash
# chỉ chạy biến thể EEG + tuổi cho nhanh
python train.py --data-root "/path/to/CauEEG" --variants age

# đổi số epoch / batch / learning rate
python train.py --data-root "/path/to/CauEEG" --epochs 30 --batch-size 64 --learning-rate 3e-4

# dùng lại cache .npy từ lần chạy trước (tên file <id>.edf.128.npy)
python train.py --data-root "/path/to/CauEEG" --cache-dir "/path/to/cache_cu"

# hiện đồ thị trên màn hình (mặc định chỉ lưu PNG)
python train.py --data-root "/path/to/CauEEG" --show
```

Mỗi biến thể mất khoảng 35-40 phút trên GPU cỡ Kaggle (5 fold x 30 epoch); đo metric train/val mỗi epoch thêm khoảng 15-20%. Dùng `--hist-every 2` nếu muốn nhanh hơn.

## Kết quả đầu ra (trong `--output-dir`)

```text
cache/                      # cache tín hiệu .npy (nếu không dùng --cache-dir)
cv/<variant>_fold<k>.pkl    # kết quả từng fold, dùng để tiếp tục khi bị ngắt
results_summary.csv         # OOF AUC [CI], balanced acc, test AUC [CI], sens/spec
oof_scores.csv              # logit OOF từng record cho age_only / age / no_age
error_by_symptom.csv        # tỉ lệ lỗi theo nhóm bệnh
roc_and_epoch_auc.png       # ROC (OOF, test) và AUC held-out theo epoch
curves_age.png              # train-vs-val Accuracy / AUC / Loss, biến thể có tuổi
curves_no_age.png           # như trên, biến thể không tuổi
oof_diagnostics.png         # confusion matrix và phân bố logit trên OOF
```

## Cách đọc kết quả

- Số chính để báo cáo: **OOF AUC + 95% CI** và balanced accuracy ở ngưỡng chọn từ OOF. Test 57 record chỉ là kiểm tra bổ sung; khoảng tin cậy của nó rất rộng.
- Nếu `OOF AUC(EEG+tuổi) - AUC(chỉ tuổi)` có CI nằm hẳn trên 0 thì EEG thực sự đóng góp ngoài tuổi. Nếu `no_age` vẫn cao (> ~0.85) thì mô hình học được tín hiệu EEG, không chỉ tuổi.
- Đường cong train/val: train được đo bằng EMA ở chế độ eval trên 160 record train cố định (không aug, không dropout); val là fold held-out. Loss là cross-entropy mức record, không label smoothing, nên thấp hơn loss lúc train.
- Hai điểm chưa kiểm soát được: (1) file đi kèm **không có patient ID** nên cùng một bệnh nhân (ghi nhiều ngày) có thể nằm ở hai fold, làm OOF hơi lạc quan; test no-overlap không bị lỗi này. Nếu có bảng record -> bệnh nhân, truyền `group_of` vào `make_folds`. (2) Nhóm Normal và Dementia có thể khác nguồn/thiết bị/thời điểm ghi; `error_by_symptom.csv` giúp thấy lỗi tập trung ở đâu.
- Bỏ MCI nên bài toán dễ hơn benchmark 3 lớp của bài báo (74.7%); đừng so trực tiếp hai con số.

## Ghi chú

- Project được chuyển từ notebook và đã kiểm tra cú pháp; phần dữ liệu, fold, baseline tuổi, báo cáo và đồ thị đã chạy thử trên dữ liệu giả lập. Phần huấn luyện mô hình (cần PyTorch + GPU + dữ liệu CAUEEG thật) chưa được chạy trong môi trường tạo project.
- Logic giữ nguyên bản notebook v3 (cùng siêu tham số, cùng cách gộp record, cùng ngưỡng Youden trên OOF). Khác biệt duy nhất: các biến global của notebook được gom vào `RecordStore`/`Normalizer`, và đồ thị được lưu PNG thay vì chỉ hiển thị.
