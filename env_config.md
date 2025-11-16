### ros 在conda环境下兼容 (Ubuntu)



#### 1. manba 安装ros支持

mamba create -n mujo_ros python=3.11

conda activate mujo_ros



先安装mujoco



最后安装ros

由于ubuntu20 python官方是3.8，mujoco pip要求3.9及以上，通过 conda-forge 安装ros支持

conda config --env --add channels robostack-noetic
conda config --env --add channels conda-forge

通过以下命令查看conda-forge
conda config --show channels

安装ros支持插件

mamba install ros-noetic-rosbag
mamba install ros-noetic-sensor-msgs -c robostack-staging



#### 2. 使用 `roslibpy`

*“仅用于“仅订阅/发布”，支持任意 Python 版本 “*

ubuntu 

```
sudo apt-get install ros-noetic-rosbridge-server
```



conda :

pip install roslibpy



python示例：

```
import roslibpy

# 连接到本地的 rosbridge
client = roslibpy.Ros(host='localhost', port=9090)
client.run()

# 定义回调函数
def my_callback(msg):
    print(msg)

# 创建订阅者
listener = roslibpy.Topic(client, '/your/topic/name', 'std_msgs/String')
listener.subscribe(my_callback)

try:
    while True:
        pass
except KeyboardInterrupt:
    client.terminate()
```

