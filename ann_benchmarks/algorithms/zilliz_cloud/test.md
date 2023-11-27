## Install

make sure python >= 3.10

```sh
sudo pip install -r requirements.txt
sudo python install.py --algorithm zilliz_cloud
```

## Run Test

### config

`ann_benchmarks/algorithms/zilliz_cloud/config.yml`

- required
    - uri / user / password
- optional
    - skip_insert - if true, skip creating collection, inserting data and building indexes.

### test

- algorithm
    - only support zilliz_cloud
- dataset
    - sift-128-euclidean
    - gist-960-euclidean
- local
    - not run in docker
- batch
    - multi-processes
- runs
    - repeat test for each group of params (default 5, recommand 1)

```sh 
# test sift
sudo python3 run.py --algorithm zilliz_cloud --dataset sift-128-euclidean --batch --runs 1

# test gist
sudo python3 run.py --algorithm zilliz_cloud --dataset gist-960-euclidean --batch --runs 1
```