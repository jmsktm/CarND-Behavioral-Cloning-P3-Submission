echo '## Application and module versions'

echo '\n\n#### Processor:'
cat /proc/cpuinfo | grep 'model name' | uniq | cut -d ":" -f2 | awk '{$1=$1};1'

echo '\n#### Linux version:'
lsb_release -d | cut -d ":" -f2 | awk '{$1=$1};1'

echo '\n#### GPU name:'
nvidia-smi --query-gpu=name --format=csv | tail -n 1 

echo '\n#### GPU driver version:'
nvidia-smi --query-gpu=driver_version --format=csv | tail -n 1

echo '\n#### Cuda version:'
nvcc --version | tail -n -1

echo '\n#### cuDNN version:'
locate cudnn | grep "libcudnn.so." | tail -n1 | sed -r 's/^.*\.so\.//'

echo '\n#### Tensorflow version:'
python -c 'import tensorflow as tf; print(tf.__version__);'

echo '\n#### Tensorflow Keras (tf.keras) version:'
python -c 'import tensorflow as tf; print(tf.keras.__version__);'

echo '\n#### Keras version:'
python -c 'import keras; print(keras.__version__);'

echo '\n#### Python version:'
python -c 'import platform; print(platform.python_version());'

echo '\n#### pip version:'
pip --version | awk '{print $2}'

echo '\n#### GCC version:'
gcc --version | grep "gcc" | awk '{print $4}'

echo '\n#### Numpy version:'
python -c 'import numpy; print(numpy.__version__)'

echo '\n#### Pandas version:'
python -c 'import pandas; print(pandas.__version__)'

echo '\n#### Matplotlib version:'
python -c 'import matplotlib; print(matplotlib.__version__)'

echo '\n#### cv2 version:'
python -c 'import cv2; print(cv2.__version__)'  
