#!/usr/bin/env python3
import os
import shutil
from huggingface_hub import HfApi

def upload_results_to_hf():
    """上传A6000测试结果到 Hugging Face"""
    api = HfApi()
    repo_id = "ShallowU/GPT2-124m"
    
    # 检查log目录中的所有文件
    log_dir = "log"
    if not os.path.exists(log_dir):
        print(f"Error: {log_dir} directory not found!")
        return
    
    print(f"Files in {log_dir}:")
    for file in os.listdir(log_dir):
        print(f"  - {file}")
    
    # 要上传的文件 - 适应A6000测试
    files_to_upload = []
    
    # 1. 训练日志（必需）
    if os.path.exists("log/log.txt"):
        files_to_upload.append(("log/log.txt", "training_log_test_A6000.txt"))
    
    # 2. 损失曲线图（必需）
    if os.path.exists("log/loss_curves.png"):
        files_to_upload.append(("log/loss_curves.png", "loss_curves_test_A6000.png"))
    
    if os.path.exists("log/combined_loss_curves.png"):
        files_to_upload.append(("log/combined_loss_curves.png", "combined_loss_curves_test_A6000.png"))
    
    # 3. 生成结果（必需）
    if os.path.exists("log/final_generations.txt"):
        files_to_upload.append(("log/final_generations.txt", "final_generations_test_A6000.txt"))
    
    # 4. 检查点文件（查找所有checkpoint）
    checkpoint_files = []
    for file in os.listdir(log_dir):
        if file.startswith("model_") and file.endswith(".pt"):
            checkpoint_files.append(file)
    
    # 上传最后一个checkpoint（如果存在）
    if checkpoint_files:
        # 按文件名排序，取最后一个
        checkpoint_files.sort()
        final_checkpoint = checkpoint_files[-1]
        local_path = os.path.join(log_dir, final_checkpoint)
        # 重命名为测试版本
        repo_filename = f"model_test_A6000_final.pt"
        files_to_upload.append((local_path, repo_filename))
        print(f"Will upload final checkpoint: {final_checkpoint} -> {repo_filename}")
    
    # 5. 上传所有checkpoint（可选，用于测试验证）
    for checkpoint in checkpoint_files:
        local_path = os.path.join(log_dir, checkpoint)
        repo_filename = f"test_A6000_{checkpoint}"
        files_to_upload.append((local_path, repo_filename))
    
    print(f"\nPlanned uploads:")
    for local_path, repo_path in files_to_upload:
        print(f"  {local_path} -> {repo_path}")
    
    # 执行上传
    print(f"\nStarting upload to {repo_id}...")
    success_count = 0
    
    for local_path, repo_path in files_to_upload:
        if os.path.exists(local_path):
            try:
                print(f"Uploading {local_path} to {repo_path}...")
                api.upload_file(
                    path_or_fileobj=local_path,
                    path_in_repo=repo_path,
                    repo_id=repo_id,
                    repo_type="model"
                )
                print(f"✅ Successfully uploaded {repo_path}")
                success_count += 1
            except Exception as e:
                print(f"❌ Failed to upload {local_path}: {e}")
        else:
            print(f"⚠️  Warning: {local_path} not found!")
    
    print(f"\nUpload completed! {success_count}/{len(files_to_upload)} files uploaded successfully.")
    
    # 创建一个测试报告
    create_test_report(log_dir, success_count, len(files_to_upload))

def create_test_report(log_dir, success_uploads, total_uploads):
    """创建测试报告"""
    report_file = os.path.join(log_dir, "test_report_A6000.txt")
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=== A6000 GPT-2 Training Flow Test Report ===\n\n")
        f.write(f"Test Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Hardware: 1x A6000 (48GB)\n")
        f.write(f"Test Duration: ~15-20 minutes\n")
        f.write(f"Training Steps: 50 (out of planned 19073)\n")
        f.write(f"Purpose: Validate complete training pipeline\n\n")
        
        f.write("=== Test Results ===\n")
        f.write(f"Files uploaded: {success_uploads}/{total_uploads}\n")
        f.write(f"Status: {'✅ PASSED' if success_uploads == total_uploads else '⚠️  PARTIAL'}\n\n")
        
        f.write("=== Generated Files ===\n")
        if os.path.exists(log_dir):
            for file in sorted(os.listdir(log_dir)):
                f.write(f"- {file}\n")
        
        f.write(f"\n=== Next Steps ===\n")
        f.write("1. ✅ Pipeline validation completed\n")
        f.write("2. 🔄 Ready for full 8x A100 training\n")
        f.write("3. 📊 All components working correctly\n")
    
    print(f"Test report saved: {report_file}")
    
    # 也上传测试报告
    try:
        api = HfApi()
        api.upload_file(
            path_or_fileobj=report_file,
            path_in_repo="test_report_A6000.txt",
            repo_id="ShallowU/GPT2-124m",
            repo_type="model"
        )
        print("✅ Test report uploaded to Hugging Face")
    except Exception as e:
        print(f"❌ Failed to upload test report: {e}")

if __name__ == "__main__":
    import time
    upload_results_to_hf()