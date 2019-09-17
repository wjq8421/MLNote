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

#### Image Processing

##### translation

```python
image = cv2.imread("Koala.jpg")
cv2.imshow("Original", image)

# 25: 向左或向右平移的像素值
# 50: 向上或向下平移的像素值
M = np.float32([[1, 0, 25], [0, 1, 50]]) # translation matrix
# 第三个参数：图像的dimension：width*height
shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0])) 
cv2.imshow("Shifted Down and Right", shifted)
cv2.waitKey(0)


def translate(image, x, y):
    # x: the number of pixels that we are going to shift along the x-axis
    # y: the number of pixels that we are going to shift along the y-axis
    M = np.float32([[1, 0, x], [0, 1, y]])
    shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    return shifted
  
shifted = translate(image, 0, 100)
cv2.imshow("Shifted Down", shifted)
cv2.waitKey(0)
```

##### rotation

```python
(h, w) = image.shape[:2]
center = (w//2, h//2)

# 沿着center旋转
# 第三个参数：the scale of the image
M = cv2.getRotationMatrix2D(center, 45, 0.5) # 45：逆时针旋转45度
rotated = cv2.warpAffine(image, M, (w, h))
cv2.imshow("Rotated by 45 Degrees", rotated)
cv2.waitKey(0)

def rotate(image ,angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]
    
    if center is None:
        center = (w//2, h//2)
    
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

rotated = rotate(image, 180)
cv2.imshow("Rotated by 180 Degrees", rotated)
cv2.waitKey(0)
```

##### resize

```python
r = 150.0 / image.shape[1]
dim = (150, int(image.shape[0] * r)) # 记住the aspect ratio of the image
resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
cv2.imshow("Resized (Width)", resized)
cv2.waitKey(0)

def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    h, w = image.shape[:2]
    
    if width is None and height is None:
        return image
    
    if width is None:
        r = height / float(h)
        dim = (int(w*r), height)
    else:
        r = width / float(w)
        dim = (width, int(h*r))
    
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized
```

##### flip

```python
flipped = cv2.flip(image, 1)
cv2.imshow("Flipped Horizontally", flipped)
cv2.waitKey(0)

flipped = cv2.flip(image, 0)
cv2.imshow("Flipped Vertically", flipped)
cv2.waitKey(0)

flipped = cv2.flip(image, -1)
cv2.imshow("Flipped Horizontally & Vertically", flipped)
cv2.waitKey(0)
```

##### arithmetic

```python
M = np.ones(image.shape, dtype='uint8') * 100
added = cv2.add(image, M)
cv2.imshow("Added", added)
cv2.waitKey(0)

M = np.ones(image.shape, dtype='uint8') * 50
subtracted = cv2.subtract(image, M)
cv2.imshow("Subtracted", subtracted)
cv2.waitKey(0)
```

##### bitwise operation

```python
rectangle = np.zeros((300, 300), dtype='uint8')
cv2.rectangle(rectangle, (25, 25), (275, 275), 255, -1)

circle = np.zeros((300, 300), dtype='uint8')
cv2.circle(circle, (150, 150), 150, 255, -1)

bitwiseAnd = cv2.bitwise_and(rectangle, circle)
cv2.imshow("AND", bitwiseAnd)
cv2.waitKey(0)
```

##### masking

> allow us to focus only on the portions of the image that interests us.

```python
mask = np.zeros(image.shape[:2], dtype='uint8')
(cX, cY) = (image.shape[1]//2, image.shape[0]//2)
cv2.rectangle(mask, (cX-75, cY-75), (cX+75, cY+75), 255, -1)
cv2.imshow("Mask", mask)

# By supplying a mask, only examines pixels that are "on" in the mask
masked = cv2.bitwise_and(image, image, mask=mask)
cv2.imshow("Mask Applied to Image", masked)
cv2.waitKey(0)
```

##### splitting and merging

```python
B, G, R = cv2.split(image)
# 图片中哪种颜色越多就越亮
cv2.imshow("Red", R)
cv2.imshow("Green", G)
cv2.imshow("Blue", B)

# 只呈现Blue颜色
zeros = np.zeros(image.shape[:2], dtype='uint8')
cv2.imshow("Blue", cv2.merge([B, zeros, zeros]))
cv2.waitKey(0)
```

ROIs（Regions of Interest）

Lab color space

HSV color space

________

#### histogram

> represent the distribution of pixel intensities. 

1. Compare histograms for similarity: compare the query image to a dataset of images, ranking the results by similarity.

```python
image = cv2.imread("Desert.jpg")

# 画三个通道的histogram
chans = cv2.split(image)
colors = ('b', 'g', 'r')
plt.figure()
plt.title("'Flattened' Color Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")

for (chan, color) in zip(chans, colors):
    hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
    plt.plot(hist, color=color)
    plt.xlim([0, 256])
         
  
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(131)
hist = cv2.calcHist([chans[1], chans[0]], [0, 1], None, [16, 16], [0, 256, 0, 256])
p = ax.imshow(hist, interpolation='nearest')
ax.set_title("2D color histogram for G and B")
plt.colorbar(p)

ax = fig.add_subplot(132)
hist = cv2.calcHist([chans[1], chans[2]], [0, 1], None, [16, 16], [0, 256, 0, 256])
p = ax.imshow(hist, interpolation='nearest')
ax.set_title("2D color histogram for G and R")
plt.colorbar(p)

ax = fig.add_subplot(133)
hist = cv2.calcHist([chans[0], chans[2]], [0, 1], None, [16, 16], [0, 256, 0, 256])
p = ax.imshow(hist, interpolation='nearest')
ax.set_title("2D color histogram for B and R")
plt.colorbar(p)

print("2D histogram shape: {}, with {} values".format(hist.shape, hist.flatten().shape[0]))
```

直方图均衡（histogram equalization）

```python
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
eq = cv2.equalizeHist(image)
cv2.imshow("Histogram Equalization", np.hstack([image, eq]))
cv2.waitKey(0)
```

LBPs（Local Binary Patterns）

Image Descriptors: Hu Moments, Zernike Moments