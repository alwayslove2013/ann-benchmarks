from multiprocessing.pool import ThreadPool
import time
import traceback
import numpy as np
from pymilvus import Collection, utility, connections, CollectionSchema, DataType, FieldSchema, MilvusException
import concurrent
import multiprocessing as mp
import numpy as np
import json

MILVUS_LOAD_REQS_SIZE = 1.5 * 1024 * 1024
METRIC_MAPPING = {"dot": "IP", "angular": "COSINE", "euclidean": "L2"}

configs = {
    "uri": "",
    "user": "",
    "password": "",
    "skip_insert": False,  # skip creating collection, inserting data and building indexes
    "detailed_batch_logs": False,
}


class ZillizCloud:
    def __init__(self, metric: str, dim: int, configs: dict):
        self.collection_name = "ann_test_collection"
        self._vector_field = "vector"
        self._index_name = "index"
        self.concurrencies = 1

        self.dim = dim
        self.metric = metric
        self.metric_type = METRIC_MAPPING.get(self.metric, None)
        assert self.metric_type is not None

        self.configs = configs
        uri = configs.get("uri", "")
        user = configs.get("user", "")
        password = configs.get("password", "")
        self.db_config = dict(uri=uri, user=user, password=password)
        self.skip_insert = configs.get("skip_insert", False)
        self.detailed_batch_logs = configs.get("detailed_batch_logs", True)
        self._connect()

        if not self.skip_insert:
            if utility.has_collection(self.collection_name):
                print("collection exsited, drop it")
                utility.drop_collection(self.collection_name)

            fields = [
                FieldSchema("primary", DataType.INT64, is_primary=True),
                FieldSchema(self._vector_field, DataType.FLOAT_VECTOR, dim=dim),
                FieldSchema("rank", DataType.INT64),
                FieldSchema("code", DataType.VARCHAR, max_length=10),
            ]
            print("create collection")
            self.col = Collection(
                name=self.collection_name,
                schema=CollectionSchema(fields),
                consistency_level="Session",
            )
            print("create index")
            self._create_index()

        self.col = Collection(self.collection_name)

    def _connect(self) -> None:
        connections.connect(**self.db_config, timeout=60)

    def _create_index(self):
        index_params = {"metric_type": self.metric_type, "index_type": "AUTOINDEX"}
        self.col.create_index(
            self._vector_field,
            index_params,
            index_name=self._index_name,
        )

    def _optimize(self):
        self.col.flush()
        self._create_index()
        utility.wait_for_index_building_complete(self.collection_name)

        def wait_index():
            while True:
                progress = utility.index_building_progress(self.collection_name)
                if progress.get("pending_index_rows", -1) == 0:
                    break
                time.sleep(5)

        wait_index()
        self.col.compact()
        self.col.wait_for_compaction_completed()
        wait_index()

    def fit(self, X: np.array, attrs: list[dict]):
        print("train", X.shape)

        if not self.skip_insert:
            print("insert")
            start = time.perf_counter()
            batch_size = int(MILVUS_LOAD_REQS_SIZE / (self.dim * 4))
            for batch_start_offset in range(0, X.shape[0], batch_size):
                batch_end_offset = min(batch_start_offset + batch_size, X.shape[0])
                attrList = attrs[batch_start_offset:batch_end_offset]
                ranks = [attr["rank"] for attr in attrList]
                codes = [attr["code"] for attr in attrList]
                insert_data = [
                    list(range(batch_start_offset, batch_end_offset)),
                    X[batch_start_offset:batch_end_offset],
                    ranks,
                    codes,
                ]
                self.col.insert(insert_data)
            cost = time.perf_counter() - start
            print(f"insert cost {cost:.3f}s")

            print("optimize")
            start = time.perf_counter()
            self._optimize()
            cost = time.perf_counter() - start
            print(f"optimize cost {cost:.3f}s")

            print("load")
            start = time.perf_counter()
            self.col.load()
            cost = time.perf_counter() - start
            print(f"load cost {cost:.3f}s")

    def set_query_arguments(self, concurrencies):
        # concurrencies for batch mode;
        # in single-query mode, nothing would happen, except repeat test;
        self.concurrencies = concurrencies

    def query(self, q: np.array, n: int, filter_expr: str):
        search_params = {"metric_type": self.metric_type}
        res = self.col.search(
            data=[q],
            anns_field=self._vector_field,
            param=search_params,
            limit=n,
            expr=filter_expr,
        )
        # ret = [result.id for result in res[0]]
        # return ret

    def batch_query(self, X: np.array, n: int):
        print(f"[Defaul ThreadPool] batch query - concurrencies: {self.concurrencies}")
        pool = ThreadPool(self.concurrencies)
        self.res = pool.map(lambda q: self.query(q, n), X)

    def __str__(self):
        return f"zilliz_cloud_{self.concurrencies}"

    def custom_batch_query(self, X: np.array, n: int, filter_exprs: list[str]):
        test_data = X.tolist()
        duration = 30
        conc = self.concurrencies
        try:
            with mp.Manager() as m:
                q, cond = m.Queue(), m.Condition()
                with concurrent.futures.ProcessPoolExecutor(
                    mp_context=mp.get_context("spawn"), max_workers=conc
                ) as executor:
                    print(f"[Custom Multiprocesses (Spawn)] start search {duration}s in concurrency {conc}")
                    configs = self.configs
                    configs["skip_insert"] = True
                    params = dict(configs=configs, metric=self.metric, dim=self.dim)
                    future_iter = [
                        executor.submit(custom_search_subtask, params, test_data, filter_exprs, duration, n, q, cond)
                        for _ in range(conc)
                    ]
                    # Sync all processes
                    while q.qsize() < conc:
                        time.sleep(3)
                    with cond:
                        cond.notify_all()
                        if self.detailed_batch_logs:
                            print("Sync all processes. [done]")

                    start = time.perf_counter()
                    all_count = sum([r.result()[0] for r in future_iter])
                    cost = time.perf_counter() - start
                    qps = round(all_count / cost, 4)
                    latency = cost * conc / all_count
                    print(f"[Custom Multiprocesses (Spawn)] all_count: {all_count}, cost: {cost:.3f}s, qps: {qps:.3f}, latency: {latency:.3f}")
        except BaseException as e:
            print(f"multiprocess error: {e}")


def custom_search_subtask(
    params: dict, test_data: list[list[float]], filter_exprs, duration: int, n: int, q: mp.Queue, cond: mp.Condition
):
    # Sync processes
    q.put(1)
    with cond:
        cond.wait()

    client = ZillizCloud(**params)
    num, idx = len(test_data), 0
    count = 0
    start_time = time.perf_counter()

    while time.perf_counter() < start_time + duration:
        try:
            client.query(test_data[idx], n, filter_exprs[idx])
        except Exception as e:
            print(f"VectorDB search_embedding error: {e}")
            traceback.print_exc(chain=True)
            raise e from None

        count += 1
        # loop through the test data
        idx = idx + 1 if idx < num - 1 else 0

    total_dur = round(time.perf_counter() - start_time, 4)
    if client.detailed_batch_logs:
        print(
            f"{mp.current_process().name:16} search {duration}s: "
            f"actual_dur={total_dur}s, count={count}, qps in this process: {round(count / total_dur, 4):3}"
        )

    return (count, total_dur)


def open_jsonl(file):
    with open(file, "r") as json_file:
        json_list = list(json_file)

    data = [json.loads(json_str) for json_str in json_list]
    return data


def insert():
    train_data = np.load("./sift10m-filter/vectors.npy")
    attrs = open_jsonl("./sift10m-filter/attrs.jsonl")
    dim = train_data.shape[1]
    client = ZillizCloud(metric="euclidean", dim=dim, configs=configs)
    client.fit(train_data, attrs=attrs)


def serial_test():
    print("serial_test")
    test_data = open_jsonl("./sift10m-filter/tests.jsonl")
    queries = np.array([json.loads(d["query"]) for d in test_data])
    filter_exprs = [d["filter"] for d in test_data]
    _configs = {**configs, "skip_insert": True}
    client = ZillizCloud(metric="euclidean", dim=128, configs=_configs)

    for k in [10, 100]:
        print(f"====> topk = {k}")
        start_time = time.perf_counter()
        for idx in range(len(queries)):
            client.query(queries[idx], k, filter_exprs[idx])
        total_dur = time.perf_counter() - start_time
        qps = len(queries) / total_dur
        latency = total_dur / len(queries)
        print(f"qps: {qps}, latency: {latency}")


def conc_test():
    print("conc_test")
    test_data = open_jsonl("./sift10m-filter/tests.jsonl")
    queries = np.array([json.loads(d["query"]) for d in test_data])
    filter_exprs = [d["filter"] for d in test_data]
    _configs = {**configs, "skip_insert": True}
    client = ZillizCloud(metric="euclidean", dim=128, configs=_configs)

    for k in [10, 100]:
        print(f"====> topk = {k}")
        for concurries in [1, 8, 32, 64, 96, 128]:
            client.set_query_arguments(concurries)
            client.custom_batch_query(queries, k, filter_exprs)


def main():
    insert()
    serial_test()
    conc_test()


if __name__ == "__main__":
    main()
