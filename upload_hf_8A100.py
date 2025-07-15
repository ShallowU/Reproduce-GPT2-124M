#!/usr/bin/env python3
# 上传训练结果到 Hugging Face
# 该脚本将上传训练好的模型、日志、损失曲线等文件
import os
import shutil
from huggingface_hub import HfApi

def upload_results_to_hf():
    """上传训练结果到 Hugging Face"""
    api = HfApi()
    repo_id = "ShallowU/GPT2-124M"
    
    # 要上传的文件
    files_to_upload = [
        ("log/model_19073.pt", "model_19073.pt"),  # 最终模型
        ("log/log.txt", "training_log.txt"),       # 训练日志
        ("log/loss_curves.png", "loss_curves.png"), # 损失曲线
        ("log/combined_loss_curves.png", "combined_loss_curves.png"),
        ("log/final_generations.txt", "final_generations.txt") , # 生成结果
        ("log/timing_summary.txt", "timing_summary.txt"), # 时间统计
    ]
    
    for local_path, repo_path in files_to_upload:
        if os.path.exists(local_path):
            print(f"Uploading {local_path} to {repo_path}")
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=repo_path,
                repo_id=repo_id,
                repo_type="model"
            )
        else:
            print(f"Warning: {local_path} not found!")
    
    print("Upload completed!")

if __name__ == "__main__":
    upload_results_to_hf()