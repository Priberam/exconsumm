# ExCoNSumm :Jointly Extracting and Compressing Documentswith Summary State Representations

This repository releases our code for ExConSum. It uses dynet and the code is in C++.

Please contact afonso@priberam.com or sebastiao@priberam.com for any question.

### Data

#### CNN and Dailymail data

The datasets and the word emebeddings are the same as for REFRESH except for the compressed version. You can get them from https://github.com/EdinburghNLP/Refresh.

The Compressive Oracles dataset used in the paper are available at:

```
wget -r ftp://"ftp.priberam.pt|anonymous"@ftp.priberam.pt/SUMMAPublic/Corpora/Exconsumm/CompressiveOracles/2019.0
```

### Training

./exconsumm -ccnndm.cfg -t --dynet-autobatch 1 --dynet-mem 7500

### Evaluation

./exconsumm -ccnndm.cfg -t --dynet-autobatch 1 --dynet-mem 7500

### Build the system

Install dynet prerequisites eigen and MKL check https://dynet.readthedocs.io/en/latest/install.html.

The first time you clone the repository, you need to sync the dynet/ submodule.

Select the backend eiher "cuda" for GPU or "eigen" for CPU, on CPU you should use MKL as described in the dynet instalation page above.

```
git submodule init
git submodule update

mkdir build
cd build
cmake .. -DEIGEN3_INCLUDE_DIR=/path/to/eigen -DBACKEND=cuda|eigen
make -j2
```




System described in the paper:

Jointly Extracting and Compressing Documents with Summary State Representations, Afonso Mendes, Shashi Narayan, Sebastião Miranda, Zita Marinho, André F. T. Martins and Shay B. Cohen (NAACL 2019, accepted, to appear). Also at arXiV: https://arxiv.org/abs/1904.02020


