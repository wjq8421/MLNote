Facial Recognition、Build representation of our 3D world、surveillance

numpy：express images as multi-dimensional arrays.

opencv：real-time image processing.



#### Load an image

```python
import cv2

image = cv2.imread("Koala.jpg") # BGR格式的图片
# height: # of rows; width: # of columns
image.shape  # 维度依次为：height, width, channels，而像素尺寸表示为width * height

cv2.imshow("Koala", image)
cv2.waitKey(0) # pause the execution until we press a key on our keyboard
cv2.imwrite("Koala_1.png", image)
```

​	`plt.imshow()`： only works for images, not suitable for displaying frames from a video stream or video file.

```python
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# image = mpimg.imread("Koala.jpg")
image = plt.imread("Koala.jpg") # RGB格式的图片
plt.axis("off")  # 取消numbered axes
plt.imshow(image)
plt.show()

# cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR转换成RGB的图片
```

________

