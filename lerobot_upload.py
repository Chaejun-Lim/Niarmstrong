from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# 1. 로컬 데이터셋 로드 (로컬 폴더 경로 지정)
dataset = LeRobotDataset(
    root="/home/jun/NO_SOL/hybrid/recordings/soliscute/NiArmStrong_0828_1", # 로컬 데이터셋 폴더 경로
    repo_id="soliscute/NiArmStrong_0828_1"  # 허브에 업로드할 데이터셋 이름,
)

# 2. 허브로 업로드 실행
dataset.push_to_hub(
    repo_id="soliscute/NiArmStrong_0828_1",
    private=False,  # 비공개 데이터셋은 True, 공개는 False
    license="apache-2.0",
)