import numpy as np
import cv2

# 读取 bin 文件
with open('/home/mamager/Documents/received_image_data.bin', 'rb') as f:
    image_data = f.read()

# 将字节数据转换为 numpy 数组
image_data = np.frombuffer(image_data, dtype=np.uint8)

# 尝试解码图像数据
image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

# 检查图像是否解码成功
if image is not None:
    # 显示解码后的图像
    cv2.imshow("Decoded Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Failed to decode image")

# 如果要生成视频流，可以按照如下步骤处理多帧图像
# 假设我们有多个图像帧数据
# frame_list = [image1, image2, image3, ...]
# 这里以示例图像列表代替实际数据
frame_list = [image] * 10  # 假设有10帧相同的图像

# 设置视频编码格式
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_writer = cv2.VideoWriter('output_video.avi', fourcc, 20.0, (image.shape[1], image.shape[0]))

# 写入帧到视频
for frame in frame_list:
    video_writer.write(frame)

# 释放视频写入对象
video_writer.release()
print("Video saved as output_video.avi")