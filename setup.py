import setuptools

setuptools.setup(
    name="stubbed",
    version="0.1.1",
    packages=setuptools.find_packages(),
    author='Stanford Plasticine Project',
    license='MIT',
    description='A library for obtaining runtime traces',
    install_requires=[
        'numpy',
        'scipy',
        'networkx'
    ]
)
