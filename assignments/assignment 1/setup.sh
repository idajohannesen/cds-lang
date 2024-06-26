# create virtual env
python -m venv env_assignment
# activate env
source ./env_assignment/bin/activate
# install requirements
pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download en_core_web_md