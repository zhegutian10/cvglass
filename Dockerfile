# OpenGlass 盲人导航系统 - 生产环境Docker镜像
# Python版本: 3.11 (推荐)
FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TZ=Asia/Shanghai

# 更换为国内镜像源
RUN rm -f /etc/apt/sources.list.d/debian.sources && \
    echo "deb http://mirrors.aliyun.com/debian/ bookworm main non-free non-free-firmware contrib" > /etc/apt/sources.list \
    && echo "deb http://mirrors.aliyun.com/debian/ bookworm-updates main non-free non-free-firmware contrib" >> /etc/apt/sources.list \
    && echo "deb http://mirrors.aliyun.com/debian/ bookworm-backports main non-free non-free-firmware contrib" >> /etc/apt/sources.list \
    && echo "deb http://mirrors.aliyun.com/debian-security bookworm-security main non-free non-free-firmware contrib" >> /etc/apt/sources.list

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    # OpenCV依赖
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    # pyttsx3依赖 (Linux TTS)
    espeak \
    espeak-data \
    libespeak1 \
    libespeak-dev \
    # 网络工具
    curl \
    # 时区数据
    tzdata \
    # 清理缓存
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# 设置时区
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# 复制依赖文件
COPY requirements.txt .

# 使用国内pip镜像源
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目文件 (排除test目录)
COPY main.py .
COPY .env .
COPY modules/ modules/
COPY music/ music/
COPY mobile/ mobile/
COPY cert.pem key.pem ./

# 可选：如果使用YOLO模型，取消下面的注释
# COPY models/ models/

# 创建数据目录
RUN mkdir -p /app/data

# 暴露端口
EXPOSE 8081

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -k -f https://localhost:8081/ || exit 1

# 启动命令
CMD ["python", "main.py"]