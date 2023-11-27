from multiprocessing.pool import ThreadPool
import time
import numpy as np
from pymilvus import Collection, utility, connections, CollectionSchema, DataType, FieldSchema
from ..base.module import BaseANN

MILVUS_LOAD_REQS_SIZE = 1.5 * 1024 * 1024
METRIC_MAPPING = {"dot": "IP", "angular": "COSINE", "euclidean": "L2"}


class ZillizCloud(BaseANN):
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
        self.skip_custom_batch_test = configs.get("skip_custom_batch_test", False)
        self.detailed_batch_logs = configs.get("detailed_batch_logs", True)
        self._connect()

        if not self.skip_insert:
            if utility.has_collection(self.collection_name):
                print("collection exsited, drop it")
                utility.drop_collection(self.collection_name)

            fields = [
                FieldSchema("primary", DataType.INT64, is_primary=True),
                FieldSchema(self._vector_field, DataType.FLOAT_VECTOR, dim=dim),
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

    def fit(self, X: np.array):
        print("train", X.shape)

        if not self.skip_insert:
            print("insert")
            start = time.perf_counter()
            batch_size = int(MILVUS_LOAD_REQS_SIZE / (self.dim * 4))
            for batch_start_offset in range(0, X.shape[0], batch_size):
                batch_end_offset = min(batch_start_offset + batch_size, X.shape[0])
                insert_data = [
                    list(range(batch_start_offset, batch_end_offset)),
                    X[batch_start_offset:batch_end_offset],
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

    def query(self, q: np.array, n: int):
        search_params = {"metric_type": self.metric_type}
        res = self.col.search(
            data=[q],
            anns_field=self._vector_field,
            param=search_params,
            limit=n,
        )
        ret = [result.id for result in res[0]]
        return ret

    def batch_query(self, X: np.array, n: int):
        print(f"[Defaul ThreadPool] batch query - concurrencies: {self.concurrencies}")
        pool = ThreadPool(self.concurrencies)
        self.res = pool.map(lambda q: self.query(q, n), X)

    def __str__(self):
        return f"zilliz_cloud_{self.concurrencies}"
