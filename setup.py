from setuptools import find_packages, setup

setup(name='dsutils',
      version='0.1',
      description='Basic utilities for ian data science',
      url='https://github.com/brendanhasz/DSUtils',
      author='ianlim',
      author_email='ianlim28@hotmail.com',
      license='MIT',
      packages=find_packages(where="src"),
      package_dir={"": "src"},
      zip_safe=False)