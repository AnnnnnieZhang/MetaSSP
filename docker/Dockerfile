# 使用官方的 Miniconda 镜像作为基础镜像
FROM continuumio/miniconda3

# 设置工作目录
WORKDIR /app

# # 将 Conda 环境文件复制到镜像中
# COPY /home/data/luoxi/SSR/environment.yml /app/environment.yml

# # 创建 Conda 环境
# RUN conda env create -f environment.yml

# # 激活 Conda 环境
# SHELL ["conda", "run", "-n", "ssr", "/bin/bash", "-c"]

# # 将你的脚本复制到镜像中
# COPY external/build_gaps.sh /app/build_gaps.sh

# # 赋予脚本执行权限
# RUN chmod +x /app/build_gaps.sh

# # 运行脚本
# CMD ["conda", "run", "-n", "ssr", "bash", "build_gaps.sh"]
