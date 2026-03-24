pip install seaborn
pip install ml-collections

# install the adversarial training package

pip install torchtyping==0.1.4
# pip install typeguard==2.11.1
pip install tinydb
pip install cvxpy
pip install evotorch
pip install sacred
pip install ogb
pip install wandb
pip install gmpy2
pip install statsmodels
pip install sympy

cd GNNSetup && python setup.py develop && cd ..
cd Eva && python setup.py develop && cd ..

pip install ml-collections==0.1.1