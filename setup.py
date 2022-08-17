from distutils.core import setup
import setuptools
setup(
    name='FactorTest',
    version='1.0.7',
    description='zxy 框架',
    long_description='详见https://github.com/jltxzxy/FactorTest.git',
    author='zhangxaingyu',
    author_email='376184494@qq.com',
    py_modules=['FactorTest','FactorTestBox','FactorTestMain','FactorTestPara','FactorTestPerformance'],
    url='https://github.com/jltxzxy/FactorTest.git',
    license='MIT',
    packages=setuptools.find_packages(),
    requires=[]
)