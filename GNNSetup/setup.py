import setuptools


# with open('README.md', 'r') as fh:
#     long_description = fh.read()


setuptools.setup(
    name="gnn_setup",
    version="0.0.1",

    author='NobodyATM',
    author_email='noemail@nowhere.hahaha',
    url='',

    description="Basic tools for running the GNN experiments.",


    packages=setuptools.find_packages(),

    classifiers=[
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],
    license='GNU License',
    # TODO: add ogb to the requirements
)