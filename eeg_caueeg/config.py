from dataclasses import dataclass

CHANNELS_19 = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4', 'T4',
               'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2']
ALIAS = {'T7': 'T3', 'T8': 'T4', 'P7': 'T5', 'P8': 'T6'}

AGE_MEAN, AGE_STD = 70.0, 10.0
AD_SYMPTOMS = {'ad', 'load', 'eoad'}          # chỉ dùng khi --ad-only
VARIANTS = ('age', 'no_age')                  # 'age' = EEG + tuổi, 'no_age' = chỉ EEG
CLASS_NAMES = ['Bình thường (HC)', 'Alzheimer (AD)']   # nhãn 0 = HC, 1 = AD/dementia


@dataclass
class TrainConfig:
    # tín hiệu
    sfreq: int = 128
    win_sec: int = 10
    skip_sec: int = 10
    windows_per_record: int = 16
    eval_windows: int = 24
    # huấn luyện
    batch_size: int = 64
    eval_batch_size: int = 128
    epochs: int = 30
    lr: float = 3e-4
    weight_decay: float = 5e-2
    clip: float = 8.0
    grad_clip: float = 1.0
    log_every: int = 3
    dropout: float = 0.25
    base_ch: int = 32
    num_workers: int = 0
    ema_decay: float = 0.998
    label_smooth: float = 0.05
    class_weight: bool = False
    aug_scale: float = 0.05
    aug_noise: float = 0.01
    # lịch sử để vẽ đường cong train/val
    hist_every: int = 1          # ghi lịch sử mỗi N epoch
    curve_windows: int = 12      # số cửa sổ/record khi đánh giá để vẽ đường cong
    train_eval_n: int = 160      # số record train cố định dùng để đo train metric

    @property
    def win(self):
        return self.sfreq * self.win_sec

    @property
    def skip(self):
        return self.sfreq * self.skip_sec
