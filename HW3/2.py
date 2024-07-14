import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# 读取两张图像
image1 = mpimg.imread('plot.png')
image2 = mpimg.imread('plot_dropout_0.4.png')

# 创建一个包含两个子图的画布
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# 显示第一张图像
ax1.imshow(image1)
ax1.set_title('plot')

# 显示第二张图像
ax2.imshow(image2)
ax2.set_title('plot_dropout_0.4')

# 设置图像标题和标签
plt.suptitle('Comparison of Two Ways')
plt.savefig('compare.png')
plt.show()
