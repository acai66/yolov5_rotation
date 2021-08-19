# convert xml labels to yolov5-rotation format.

- you can label images with following tools:
1. [labelimg2](https://github.com/chinakook/labelImg2)
2. [rolabelimg](https://github.com/cgvict/roLabelImg)

- converted label format:

```
classid center_x center_y longside shortside angle
```

center_x, longside are divided by image width.

center_y, shortside are divided by image height.

angle is the angle between the long side and the x-axis, the range is [0, 180). The x-axis rotate counterclockwise by angle will parallel to the long side


# 转换数据标注格式

- 旋转框标注可以使用如下标注软件:
1. [labelimg2](https://github.com/chinakook/labelImg2)
2. [rolabelimg](https://github.com/cgvict/roLabelImg)

- 转换后标签格式:

```
classid center_x center_y longside shortside angle
```

center_x, longside 是相对于图像宽度的比例

center_y, shortside 是相对于图像高度的比例

angle 是 长边 与 x 轴之间的夹角，范围为[0, 180)，x轴逆时针方向旋转angle度即与长边平行。

