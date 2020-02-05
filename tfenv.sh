module load python/3.6
virtualenv --no-download my_tf_env
source my_tf_env/bin/activate
pip install tensorflow_gpu==2.0 --no-index
