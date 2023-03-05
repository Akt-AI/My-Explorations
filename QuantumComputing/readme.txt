sudo apt install virtualenv 
virtualenv -p python2.7 env2
source env2/bin/activate 
pip install -r requirements.txt

python plot_wigner_function.py
