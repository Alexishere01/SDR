from setuptools import setup, find_packages

setup(
    name='geminisdr',
    version='0.1.0',
    description='A general-purpose SDR toolkit for researching interference mitigation techniques in drone mesh networks.',
    author='Gemini',
    author_email='',
    url='',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'matplotlib',
        'torch',
        'scipy',
        'h5py',
        'scikit-learn',
        'psutil',
        'mlflow',
        'pyadi-iio',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
