class FacialDetConfig:
    IMG_WIDTH: int = 96
    IMG_HEIGHT: int = 96

    IN_CHANNELS: int = 1
    K_CHANNELS: list[int] = [8, 16, 32, 16]
    LINEAR_DIMS: list[int] = [512, 256, 32]

    OUTPUT_DIM: int = 8


class FacialDetTrainerConfig:
    DATASET_PATH: str = "../datasets/face_images.npz"
    DATASET_LABELS_PATH: str = "../datasets/facial_keypoints.csv"

    ISOLATED_POINTS: list[int] = [0, 1, 2, 3, 20, 21, 28, 29]

    EPOCH_COUNT: int = 15

    BATCH_SIZE: int = 16

    TRAINING_PERCENT: float = 0.9
