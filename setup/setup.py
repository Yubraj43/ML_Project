from setuptools import setup, find_packages

setup(
    name='ML_Project',
    version='0.1',
    author='Yubaraj Mahato',
    author_email='mahatoyubraj43@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)