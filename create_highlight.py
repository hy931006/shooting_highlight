"""
投篮精彩集锦生成工具

根据检测结果的时间点，截取进球前后片段并拼接成集锦视频
"""

import json
import subprocess
import os
import sys
from pathlib import Path


def extract_clip(video_path: str, timestamp: float, before: float, after: float, output_path: str) -> bool:
    """
    从视频中截取指定时间点的片段

    Args:
        video_path: 原始视频路径
        timestamp: 时间点（秒）
        before: 截取前多少秒
        after: 截取后多少秒
        output_path: 输出片段路径
    """
    start_time = max(0, timestamp - before)
    duration = before + after

    cmd = [
        'ffmpeg', '-y',
        '-i', video_path,
        '-ss', str(start_time),
        '-t', str(duration),
        '-c', 'copy',
        output_path
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"  错误: {e.stderr.decode() if e.stderr else str(e)}")
        return False


def concatenate_clips(clip_paths: list, output_path: str) -> bool:
    """
    拼接多个视频片段

    Args:
        clip_paths: 片段路径列表
        output_path: 输出文件路径
    """
    # 创建临时文件列表
    list_file = output_path.replace('.mp4', '_list.txt')
    with open(list_file, 'w') as f:
        for clip in clip_paths:
            f.write(f"file '{clip}'\n")

    cmd = [
        'ffmpeg', '-y',
        '-f', 'concat',
        '-safe', '0',
        '-i', list_file,
        '-c', 'copy',
        output_path
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
        # 清理临时文件
        os.remove(list_file)
        return True
    except subprocess.CalledProcessError as e:
        print(f"  错误: {e.stderr.decode() if e.stderr else str(e)}")
        if os.path.exists(list_file):
            os.remove(list_file)
        return False


def create_highlight(
    video_path: str,
    json_path: str,
    output_path: str = 'highlight.mp4',
    before: float = 2.0,
    after: float = 1.0,
    min_confidence: float = 0.0
) -> None:
    """
    根据检测结果生成精彩集锦

    Args:
        video_path: 原始视频路径
        json_path: 检测结果JSON文件路径
        output_path: 输出文件路径
        before: 进球前截取秒数
        after: 进球后截取秒数
        min_confidence: 最小置信度阈值
    """
    # 读取检测结果
    with open(json_path, 'r') as f:
        data = json.load(f)

    # 筛选进球事件
    made_shots = [
        e for e in data['events']
        if e['type'] == 'made' and e['confidence'] >= min_confidence
    ]

    if not made_shots:
        print("没有找到进球事件")
        return

    print(f"找到 {len(made_shots)} 个进球事件")
    print(f"截取策略: 进球前 {before}s + 进球后 {after}s")
    print()

    # 创建临时片段目录
    temp_dir = Path('temp_clips')
    temp_dir.mkdir(exist_ok=True)

    clip_paths = []
    success_count = 0

    # 逐个截取片段
    for i, shot in enumerate(made_shots):
        timestamp = shot['timestamp']
        confidence = shot['confidence']
        clip_path = str(temp_dir / f'clip_{i:03d}.mp4')

        print(f"[{i+1}/{len(made_shots)}] 截取 {timestamp:.2f}s (置信度: {confidence:.2f})")

        if extract_clip(video_path, timestamp, before, after, clip_path):
            clip_paths.append(clip_path)
            success_count += 1
        else:
            print(f"  失败，跳过")

    print(f"\n成功截取 {success_count}/{len(made_shots)} 个片段")

    if not clip_paths:
        print("没有成功截取任何片段")
        return

    # 拼接片段
    print(f"拼接 {len(clip_paths)} 个片段到 {output_path}...")
    if concatenate_clips(clip_paths, output_path):
        print(f"完成! 精彩集锦已保存到: {output_path}")
    else:
        print("拼接失败")

    # 清理临时片段
    print("清理临时文件...")
    for clip in clip_paths:
        if os.path.exists(clip):
            os.remove(clip)
    if temp_dir.exists():
        temp_dir.rmdir()


def main():
    # 默认参数
    video_path = './SDIT4229.MP4'
    json_path = './results/shots_SDIT4229_v4.json'
    output_path = './highlight.mp4'
    before = 2.0  # 进球前2秒
    after = 1.0   # 进球后1秒

    # 允许命令行参数覆盖
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    if len(sys.argv) > 2:
        json_path = sys.argv[2]
    if len(sys.argv) > 3:
        output_path = sys.argv[3]

    create_highlight(video_path, json_path, output_path, before, after)


if __name__ == '__main__':
    main()
