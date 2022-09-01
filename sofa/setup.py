#!/usr/bin/env python
from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('requirements.txt') as f:
    requirements = f.read()

setup(
    # Metadata
    name='sofa',
    version='1.0.0',
    python_requires='>=3.6',
    author='AliceMind Team',
    author_email='zhicheng.zzc@alibaba-inc.com'
                 'yuze.zyz@alibaba-inc.com'
                 'suluyan.sly@alibaba-inc.com'
                 'hemu.zp@alibaba-inc.com',
    url='https://gitlab.alibaba-inc.com/groups/pretrainplatform',
    description='Supporter Of Foundation Ai',
    long_description=readme,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    license='Apache-2.0',

    #Package info
    install_requires=requirements
)
