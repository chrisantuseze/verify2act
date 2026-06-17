conda create -n verify2act python=3.10

conda activate verify2act

cd /robosuite
pip install -e .

or through Points2Plans

conda env create -f conda_env.yml

cd /robosuite
pip install -e .

# calvin
pip install -e calvin/calvin_env

pip install -e calvin/calvin_models --no-deps

pip install pytorch-lightning gym pyhash

Note: Activate env before running mjpython ...


# third_party
pip install -e third_party/MoDE_Diffusion_Policy
