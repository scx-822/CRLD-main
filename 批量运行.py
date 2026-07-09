# 这是一个批量运行实验的程序
import subprocess  # 用来运行命令
import time  # 用来暂停

# 定义要运行的所有实验
experiments = [
    # "python tools/train_proto.py --cfg configs/cifar100/crld_proto/res110_res32.yaml",
    # "python tools/train_proto.py --cfg configs/cifar100/crld_proto/wrn40_2_wrn16_2.yaml",
    # "python tools/train_proto.py --cfg configs/cifar100/crld_proto/wrn40_2_wrn40_1.yaml",
    "python tools/train_proto.py --cfg configs/cifar100/crld_proto/vgg13_vgg8.yaml"
]

print("开始批量运行实验...")

# 一个接一个运行实验
for i, command in enumerate(experiments):
    print(f"正在运行第{i + 1}个实验...")
    print(f"命令: {command}")

    # 运行命令
    subprocess.run(command, shell=True)

    print(f"第{i + 1}个实验完成！")

    # 如果不是最后一个实验，暂停一下
    if i < len(experiments) - 1:

        print("暂停10秒再运行下一个...")
        '/;.,m n b`'
        time.sleep(10)

print("所有实验都完成了！")