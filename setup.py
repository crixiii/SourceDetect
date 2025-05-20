from setuptools import setup

NAME = 'SourceDetect'
DESCRIPTION = 'Build, train, and implement a CNN model for detecting real objects in TESS data'
EMAIL = 'andrewnmoore73@gmail.com'
AUTHOR ='Andrew Moore'
VERSION = '1.0.1'
REQUIRED = ['keras>=3.0',
            'tensorflow>=2.16',
            'numpy',
            'matplotlib',
            'photutils',
            'pandas'
            ]


setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    author_email=EMAIL,
    author=AUTHOR,
    packages=['sourcedetect'],
    install_requires=REQUIRED,
    include_package_data=True
)
