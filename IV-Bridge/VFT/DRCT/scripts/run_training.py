import subprocess

def run_training():
    # 定义训练命令
    command = [
        "python", "train_contrastive.py",
        "--root_path", "/data2/lp/IVBridge_datasets/crop/train0/0real",
        "--fake_root_path", "/data2/lp/IVBridge_datasets/crop/train0/1fake",
        "--dataset_name", "GenVideo",
        "--model_name", "convnext_base_in22k",
        "--embedding_size", "1024",
        "--input_size", "224",
        "--batch_size", "64",
        "--fake_indexes", "2",
        "--num_epochs", "40",
        "--device_id", "4",
        "--lr", "0.0001",
        "--is_amp",
        "--is_crop",
        "--num_workers", "12",
        "--save_flag", "_drct_amp_crop"
        "--save_txt", "/data/lp/DRCT/output/GenVideo/result.txt"
    ]

    # 执行命令
    print("Running training command...")
    subprocess.run(command)

if __name__ == "__main__":
    run_training()