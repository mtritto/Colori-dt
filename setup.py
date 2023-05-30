from setuptools import setup
import platform

extra_require = {}
if platform.system() == 'Darwin' and platform.processor() == 'arm64':
    extra_require['apple_silicon'] = ['tensorflow-macos=2.12.0, tensorflow-metal=0.8.0']

# Read requirements.txt
with open('requirements.txt') as f:
    required = f.read().splitlines()

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='Colori-dt',
    version='0.1.0',
    author='Michele Tritto',
    author_email='m.tritto@yandex.com',
    install_requires=required,
    extra_require=extra_require,
    entry_points={
        'console_scripts': [
            'colori-dt=colori_dt.__main__:main']},
    license='MIT',  
    description=
        'Colori-dt: a python tool for color difference and neural style transfer between images', 
    long_description=long_description
)


    


