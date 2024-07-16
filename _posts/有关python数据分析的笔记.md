---
title: 有关python数据分析的笔记
date: 2022-01-22 20:27:02
tags: [收藏,Python]
---

### 1、Numpy

[numpy中文教程](https://www.yiibai.com/numpy/)

[官方文档](https://docs.scipy.org/doc/numpy-dev/user/quickstart.html)

- Numpy是一个用于进行数组运算的库
- Numpy中最重要的对象是称为ndarray的N维数组类型
- 一般使用如下语句导入:import numpy as np
- 创建数组:numpy.array(object, dtype = None, copy = True, order = None, subok = False, ndmin = 0)
- 可以用np.dtype()定义结构体
- 数组维度:ndarray.shape
- 数组维数:ndarray.ndim
- 调整数组维度:ndarray.reshape(shape)
- 创建未初始化数组:numpy.empty(shape, dtype = float, order = 'C')
- 创建零数组:numpy.zeros(shape, dtype = float, order = 'C')
- 创建一数组:numpy.ones(shape, dtype = float, order = 'C')
- 用现有数据创建数组:numpy.asarray(a, dtype = None, order = None)
- 按数值范围创建数组:numpy.arange(start = 0, stop, step = 1, dtype),类似的有linspace()和logspace()
- 切片:b=a[start:stop:step],可以用...代表剩余维度
- 整数索引:每个整数数组表示该维度的下标值,b=a[[r1, r2], [c1, c2]]
- 布尔索引:返回是布尔运算的结果的对象,可以用&或|连接()分隔的条件
- 在 NumPy 中可以对形状不相似的数组进行操作,因为它拥有广播功能,我的理解是,广播是一种维度的单方向拉伸
- 数组迭代:numpy.nditer(ndarray)或ndarray.flat
- 数组长度:len(arr)
- 访问第i个元素:一维数组用a[i],多维数组用a.flat[i]
- 数组转置:ndarray.T
- 数组分割:numpy.split(ary, indices_or_sections, axis),第二项的值为整数则表明要创建的等大小的子数组的数量,是一维数组则表明要创建新子数组的点。
- 追加值:numpy.append(arr, values, axis)
- 插入值:numpy.insert(arr, idx, values, axis)
- 删除值:numpy.delete(arr, values, axis)
- 去重数组:numpy.unique(arr, return_index, return_inverse, return_counts)
- 字符串函数:numpy.char类
- 三角函数:numpy.sin(arr),numpy.cos(arr),numpy.tan(arr)
- 四舍五入:numpy.around(arr,decimals)
- 向下取整:numpy.floor(arr)
- 向上取整:numpy.ceil(arr)
- 取倒数:numpy.reciprocal(arr),注意对于大于1的整数返回值为0
- 幂运算:numpy.power(arr,pow),pow可以是一个数,也可以是和arr对应的数组
- 取余:numpy.mod(a,b),b可以是一个数,也可以是和a对应是数组
- 最小值:numpy.amin(arr,axis)
- 最大值:numpy.amax(arr,axis)
- 数值跨度:numpy.ptp(arr,axis)
- 算术平均值:numpy.mean(arr,axis)
- 标准差:numpy.std(arr)
- 方差:numpy.var(arr)
- 副本的改变会影响原数组(赋值),视图的改变不会影响原数组(ndarray.view(),切片,ndarray.copy())
- 线性代数:numpy.linalg模块

### 2、Matplotlib

[官方教程](https://matplotlib.org/users/pyplot_tutorial.html)

[官方教程中文翻译](https://www.jianshu.com/p/c495e663f0ed)

[matplotlib入门教程](http://blog.csdn.net/wizardforcel/article/details/54407212)

[Jupyter Notebook Viewer的matplotlib lecture](http://nbviewer.jupyter.org/github/jrjohansson/scientific-python-lectures/blob/master/Lecture-4-Matplotlib.ipynb)[ ](http://nbviewer.jupyter.org/github/jrjohansson/scientific-python-lectures/blob/master/Lecture-4-Matplotlib.ipynb)

建议先看官方教程,通过折线图熟悉基本操作,然后看入门教程第三章到第六章掌握各种图的画法

 

- 一般使用如下语句导入:import matplotlib.pyplot as plt
- 绘图:[plt.plot(x,y)](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot.html#matplotlib.pyplot.plot),可选color,marker,label等参数,默认的x坐标为从0开始且与y长度相同的数组,x坐标与y坐标一般使用numpy数组,也可以用列表
- 设置线条:[plt.setp()](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.setp.html#matplotlib.pyplot.setp)
- 轴名称:plt.xlable('str'),plt.ylable('str)
- 添加文本:plt.txt(xpos,ypos,'str')
- 添加格子:plt.grid(True)
- 展示图片:plt.show()
- 图题:plt.title('str')
- 图示:plt.legend(),结合plot()中的label参数使用
- 获取子图:plt.sublot(nrows,ncols,index)或plt.subplot2grid((nrows,ncols),(rows,cols)),可选colspan和rowspan属性
- 创建画布:plt.figure()
- 数学表达式:[TeX表达式](https://matplotlib.org/users/mathtext.html#mathtext-tutorial)
- 非线性轴:plt.xscale('scale'),plt.yscale('scale'),可选参数log,symlog,logit等
- 填充颜色:plt.fill(x,y)和plt.fill_between(x,y,where=...)
- 条形图:plt.bar(x,y),注意多个条形图的默认颜色相同,应选择不同的颜色方便区分
- 直方图:plt.hist(x,bins),直方图是一种显示区段内数据数量的图像,x为数据,bins为数据区段,可选histtype,rwidth等属性
- 散点图:plt.scatter(x,y),散点图通常用于寻找相关性或分组,可选color,marker,label等属性
- 堆叠图:plt.stackplot(x,y1,y2,y3...),堆叠图用于显示部分对整体随时间的关系,通过利用plt.plot([],[],color,label)添加与堆叠图中颜色相同的空行,可以使堆叠图的意义更加清晰,可选colors等属性
- 饼图:plt.pie(slice),饼图用于显示部分对整体的关系,可选labels,colors,explode,autupct等属性

### 3、Pandas

[10 Minutes to Pandas](http://pandas.pydata.org/pandas-docs/stable/10min.html)

[十分钟搞定pandas(上文翻译版)](https://www.cnblogs.com/chaosimple/p/4153083.html)

[利用python进行数据分析](https://book.douban.com/subject/25779298/)

上面两个教程用于速成,下面这本书是pandas的作者写的,用于仔细了解

 

- 一般使用如下语句导入:import pandas as pd
- Pandas是基于NumPy 的一种工具,提供了一套名为DataFrame的数据结构,比较契合统计分析中的表结构,可用Numpy或其它方式进行计算
- 创建Series:pd.Series=(data,index),Series是一维数组
- 创建DataFrame:pd.DataFrame(data,index,colums),也可以传递一个字典结构来填充data和colums,DataFrame类似于二维表格,简称df
- 查看df头尾行:df.head(i),df.tail(i),如不填参数则分别返回除了前五行/倒数前五行的内容
- 查看索引/列/数据:df.index,df.colums,df.values
- 快速统计汇总:df.descrbe()
- 数据转置:df.T
- 按轴排序:df.sort_index(axis=0,ascending=True)
- 按值排序:df.sort_values(colums,axis=0,ascending=Ture)
- 获取:df['columnname']或df.columnname,会返回某列
- 通过条件选取某列:df = df[df('columns') == 'a']
- 对行切片:df[start:stop:step],利用df[n:n+1]即可获取某行
- 通过标签选择某行:df.loc[index,columname]
- 通过位置选择某行:df.iloc[indexpos,columnpos],df.iloc[i,:]可获取一行,df.iloc[:,i]可获取一列
- 布尔索引:df[bool],可以对单独的列进行判定,也可以对整个DataFrame进行判定
- 在pandas中使用np.nan代替缺失值,这些值不会被包含在计算中
- 对index和columns进行增删改:df.reindex(index,columns)
- 去掉含有缺失值的行:df.dropna(how='any'),可以选择how='all'只去掉所有值均缺失的行
- 补充缺失值:df.fillna(value)
- 数据应用:df.apply(func),可以是现有函数也可以是lambda函数
- 连接:pd.contact(obj),obj可以是Series,DataFrame,Panel
- 合并:pd.merge(left,right)
- 追加:df.append(data)
- 分组:df.groupby(columnname).func(),通常为分组/执行函数/组合结果
- 时间:pandas有着重采样等丰富的时间操作
- 写入CSV文件:df.to_csv(filename)
- 读取CSV文件:df.read_csv(filename),结果为DataFrame
