from setuptools import setup, find_packages

setup(
    name='TaGra',
    version='0.1.6',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'matplotlib',
        'networkx',
        'plotly',
    ],
    entry_points={
        'console_scripts': [
            'tagra=ta_gra:main',
        ],
    },
    author='Davide Torre, Davide Chicco',
    author_email='davidetorre92@gmail.com',
    description='TaGra: TAbular data preprocessing to GRAph representation.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/davidetorre92/TaGra',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
)
