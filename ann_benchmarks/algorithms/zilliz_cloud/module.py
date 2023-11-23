import time
import numpy as np
from pymilvus import Collection, utility, connections, CollectionSchema, DataType, FieldSchema, MilvusException

from ..base.module import BaseANN

MILVUS_LOAD_REQS_SIZE = 1.5 * 1024 * 1024
METRIC_MAPPING = {"dot": "IP", "angular": "COSINE", "euclidean": "L2"}


class ZillizCloud(BaseANN):
    def __init__(self, metric, dim, *wargs, **kwargs):
        self.dim = dim
        metric = METRIC_MAPPING.get(metric, None)
        assert metric is not None
        self.metric = metric

        uri = ""
        user = ""
        password = ""
        self.db_config = dict(uri=uri, user=user, password=password)
        connections.connect(**self.db_config, timeout=30)

        self.collection_name = "ann_test_collection"
        if utility.has_collection(self.collection_name):
            print("collection exsited, drop it")
            utility.drop_collection(self.collection_name)

        self._vector_field = "vector"
        self._index_name = "index"
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
        self._create_index()
        connections.disconnect("default")

    def _create_index(self):
        print("create index")
        index_params = {"metric_type": self.metric, "index_type": "AUTOINDEX"}
        self.col.create_index(
            self._vector_field,
            index_params,
            index_name=self._index_name,
        )

    def fit(self, X: np.array):
        print("train", X.shape)
        connections.connect(**self.db_config, timeout=30)
        self.col = Collection(self.collection_name)
        batch_size = int(MILVUS_LOAD_REQS_SIZE / (self.dim * 4))

        for batch_start_offset in range(0, X.shape[0], batch_size):
            batch_end_offset = min(batch_start_offset + batch_size, X.shape[0])
            insert_data = [
                list(range(batch_start_offset, batch_end_offset)),
                X[batch_start_offset:batch_end_offset],
            ]
            self.col.insert(insert_data)

        self._optimize()

    def _optimize(self):
        self.col.flush()
        self._create_index()
        print("compact")

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
        
        print("load")
        self.col.load()

    def query(self, q: np.array, n: int):
        search_params = {"metric_type": self.metric}
        res = self.col.search(
            data=[q],
            anns_field=self._vector_field,
            param=search_params,
            limit=n,
        )
        ret = [result.id for result in res[0]]
        return ret

    def __str__(self):
        return f"zilliz_cloud"