from  pymilvus  import model, MilvusClient, exceptions, DataType
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from pymilvus.model.reranker import BGERerankFunction
import time
import os
from MilvusHostError import MilvusHostError
import logging
from dotenv import load_dotenv

class MilvusUtil:
    
    def __init__(self):
        # load environment variables from .env file
        load_dotenv()
        # Basic configuration
        logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s][%(name)s]: %(message)s')
        # Creating a logger
        self.logger = logging.getLogger(__name__)
        # Milvus Server Url
        self.milvus_instance_uri = os.getenv("MILVUS_SERVER_URL")
        # Milvus user:password
        self.milvus_token = os.getenv("MILVUS_TOKEN")
        self.embedding_dimension = 1024
        self.embedding_fn = BGEM3EmbeddingFunction(
                            model_name='BAAI/bge-m3', # Specify the model name
                            device='cuda:0', # Specify the device to use, e.g., 'cpu' or 'cuda:0'
                            use_fp16=False # Specify whether to use fp16. Set to `False` if `device` is `cpu`
                            )
        self.client = self._get_milvus_client()
        self.default_output_fields = ["text", "subject"]
        self.rerank_fn = BGERerankFunction(
                            model_name="BAAI/bge-reranker-v2-m3",
                            device="cuda:0"  # Specify the device to use, e.g., 'cpu' or 'cuda:0'
                         )

    def _get_milvus_client(self):
            try:
                return MilvusClient(uri=self.milvus_instance_uri, token=self.milvus_token)
            except exceptions.MilvusException as e:
                raise MilvusHostError(self.milvus_instance_uri)

    def create_database(self, db_name):
        self.client.create_database(db_name=db_name)

    def create_schema(self):
        schema = self.client.create_schema(enable_dynamic_field=True)
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=self.embedding_dimension)
        index_params = self.client.prepare_index_params()
        index_params.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="L2")
        return schema, index_params
        
    def create_collection_milvus_db(self, collection_name="demo_collection"):
        if not self.client.has_collection(collection_name):
            schema, index_params = self.create_schema()
            self.client.create_collection(collection_name=collection_name,
                                     dimension=self.embedding_dimension,
                                     schema=schema, index_params=index_params)
            self.logger.debug(f"Collection '{collection_name}' has been created!")
        else:
            self.logger.debug(f"Collection '{collection_name}' has not been created!")

    def insert_docs_milvus_db(self, docs, collection_name="demo_collection"):
        vectors = self.embedding_fn.encode_documents(docs)['dense']
        data = [{"vector": vectors[i], "text": docs[i]} for i in range(len(vectors))]
        self.create_collection_milvus_db(collection_name)
        res = self.client.insert(collection_name=collection_name, data=data)
        self.logger.debug(res) 
                
    def search_milvus_db(self, query_text_arr, output_fields=None, collection="demo_collection"
                         , result_limit=2):
        try:
            query_vectors = self.embedding_fn.encode_queries(query_text_arr)['dense']
            if output_fields is None:
                output_fields = self.default_output_fields
            res = self.client.search(
                collection_name=collection,  # target collection
                data=query_vectors,  # query vectors
                limit=result_limit,  # number of returned entities
                output_fields=output_fields,  # specifies fields to be returned
            )

            search_results = [(ans['entity']['text'], ans['distance']) for ans in res[0]]
            return search_results
        except Exception as e:
            self.logger.critical(e)
            raise e

    def rerank(self, query, search_results):
        return self.rerank_fn(query, search_results, top_k=3)

    def search_rerank(self, query, collection="demo_collection", result_limit=3):
        start = time.time()
        search_results = self.search_milvus_db([query]
                                               , collection=collection
                                               , result_limit=5)
        search_result_texts = [result for result, score in search_results]
        rerank_results =  self.rerank_fn(query, search_result_texts, top_k=result_limit)
        self.logger.info(rerank_results)
        self.logger.info("Time taken: ", time.time() - start)
        return [result.text for result in rerank_results]


if __name__=="__main__":
    milvus_util = MilvusUtil()
    milvus_util.logger.info("Start..............")
    """
    milvus_util.insert_docs_milvus_db(["During World War II, Turing worked for the Government Code and Cypher School at Bletchley Park, Britain's codebreaking centre that produced Ultra intelligence.",
                           "Artificial intelligence was founded as an academic discipline in 1956.",
    "Alan Turing was the first person to conduct substantial research in AI.",
    "Born in Maida Vale, London, Turing was raised in southern England."], "new_collection_3" )
    milvus_util.insert_docs_milvus_db(["Alan Turing was a good human being."], "new_collection_3" )

    """

    results = milvus_util.search_milvus_db(["who is Alan Turing?"],collection="new_collection_3"
                                           , result_limit=3)
    [milvus_util.logger.info(result) for result in results]