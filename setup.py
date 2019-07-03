from distutils.core import setup

setup(
    name='PyTorchHelperBot',
    version='0.1.7',
    packages=['helperbot', ],
    install_requires=[
        'torch>=0.4.1',
        'dataclasses',
        'tqdm>=4.29.1',
        'scikit-learn>=0.21.2'
    ],
    classifiers=[  # Optional
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',

        # Pick your license as you wish
        'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
)
