from setuptools import setup, find_packages

setup(
    name='synergy-3dmm',
    version='0.0.1',
    description='Library for accurate and fast 3d landmarks, face mesh, and face pose prediction',
    packages=find_packages(exclude=('artistic*', 'benchmark*', 'loss_definition*', 'demo*', 'img*', 'pretrained*', 'Sim3DR', 'main_train*', 'singleImage*', 'model_building*', 'uv_*', 'FaceBoxes*', '3dmm_data*')),
    install_requires=[
        'torch',
        'numpy',
    ],
)