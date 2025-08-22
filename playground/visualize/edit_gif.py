from PIL import Image, ImageSequence

input_path = r"C:\Users\hsung\Videos\npp-darknpp-emoji-alpha-refresh-issue-gif.gif"
output_path = input_path  # 覆写原文件

# 读取GIF
im = Image.open(input_path)
frames = []
durations = []

# 收集帧和持续时间
for frame in ImageSequence.Iterator(im):
    frames.append(frame.copy())
    durations.append(frame.info.get('duration', 100))

# 加速2倍：每两帧取一帧，或持续时间减半
# 方法1：每两帧取一帧
# new_frames = frames[::2]
# new_durations = durations[::2]

# 方法2：持续时间减半（推荐，能保留所有帧，动画更流畅）
new_frames = frames
new_durations = [max(1, int(d/2)) for d in durations]

# 保存GIF
new_frames[0].save(
    output_path,
    save_all=True,
    append_images=new_frames[1:],
    duration=new_durations,
    loop=im.info.get('loop', 0),
    disposal=im.info.get('disposal', 2),
    transparency=im.info.get('transparency'),
)
