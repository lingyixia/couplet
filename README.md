# Usage:
 1. 4个python文件只需要动Main.py即可
 2. 108行和106行选一取消注释用以train
 3. 108行是边train边eval,但是eval会减慢总体速度,可以开始先用这个看看效果随后注释掉108，使用106行train即可(其实机器好直接用108行就行了，主要是比较穷。。。)
 4. 108行只train无eval
 5. train一段时间后把108和106都注释吊即可,112行是predict，重新运行即可看模型predict结果
# other:
 1.每120秒会保存一个模型，最多保存5个
 2. app.py是一个flask服务
 3. temp.txt是在开启flask服务器起作用的,其实就是把前端传来的上联放到里面，然后在读出来predict，因为train的时候是从文件中读的，就懒得改了，直接用的原来的函数.

# 日志记录
1. 2019.5.10 完成整体项目
2. 2019.6.28 添加attention
3. 2019.7.2 添加beamsearch
4. 2019.7.10 发现将输出层Dense的激活函数去掉收敛速度会大大提升，还不清楚原因
 - 似乎是想明白了，倒数第二层我开始用的relu激活函数，relu之后有些神经元就死了，虽然经过softmax会有一些缓解但是这些死的神经元更新依旧很慢，因此收敛很慢，结果很受初始化参数的影响(eg:"春风风去一人人"))
 因此relu虽然好，但是坚决不能用在后面，总之要用就用到前几层，离输出层远点，这样即使死了也能在后面层中复活，达到加速收敛的目的。
5. 2019.7.11 将decoder的多层lstm改为单个，用以减少参数,将encoder的多层得到的state映射一下维度,传到decoder,发现效果并没减。
下一个目标:加平仄

# 效果:
上联:天增岁月人增寿 

&emsp;下联:春满乾坤福满门 

&emsp;其他 

&emsp;春满乾坤福满堂 

&emsp;春满乾坤岁更新 

&emsp;春满乾坤地生辉 

&emsp;春满乾坤景焕新 

上联:欲穷千里目 

下联:更上一层楼

其他

不负万年心

不负一枝春

不负一生心

不负一生情


上联:春风又绿江南岸 

下联:旭日初红塞北天

其他

旭日初升塞北天

明月初圆塞北天

旭日初红塞北花

明月常明塞北天


上联:大帝君臣同骨肉 

下联:高山流水是知音

其他

中华儿女共风流

中华儿女并肩挑

中华儿女共心期

中华儿女尽心肝

目前训练20万轮作用，batchsize=32，loss=3.2左右，但是还处于下降的趋势，继续训练效果肯定会更好，但是人穷没办法，使用的google colab，虽然比本机块个几倍但是还是不够快，先这样吧
** 试用:http://49.232.34.153:7777/ 界面很丑勿喷,也没处理异常,只用于测试**
