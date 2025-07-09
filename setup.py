from setuptools import setup, find_packages
from codecs import open

with open("README.md","r",encoding='utf-8') as fh:
    long_description = fh.read()

setup(
    name="MLQuant",
    version="0.0.0",
    author="LH.Li",
    author_email="lh98lee@zju.edu.cn",  
    description='Package for ML quant',
    long_description=long_description,
    # 描述文件为md格式
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires = [
    ],
)

# python setup.py install --u
# python setup.py develop