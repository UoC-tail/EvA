import setuptools


# with open('README.md', 'r') as fh:
#     long_description = fh.read()


setuptools.setup(
    name="eva",
    version="0.0.1",

    author='NobodyATM',
    author_email='noemail@nowhere.hahaha',
    url='',

    description="Evolutionary Attack for graphs.",


    packages=setuptools.find_packages(),

    classifiers=[
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],
    license='GNU License',
    # TODO: add ogb to the requirements
)