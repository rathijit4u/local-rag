from  pymilvus  import model, MilvusClient, exceptions, DataType
import os
from MilvusHostError import MilvusHostError
import logging
from dotenv import load_dotenv

class MilvusUtil:
    
    def __init__(self):
        # load environment vairables from .env file
        load_dotenv()
        # Basic configuration
        logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s][%(name)s]: %(message)s')
        # Creating a logger
        self.logger = logging.getLogger(__name__)
        # Milvus Server Url
        self.milvus_instance_uri = os.getenv("MILVUS_SERVER_URL")
        # Milvus user:password
        self.milvus_token = os.getenv("MILVUS_TOKEN")
        self.embedding_dimention = 768
        self.embedding_fn = model.DefaultEmbeddingFunction()


    def get_milvus_client(self, milvus_instance_uri, milvus_token):
        try:
            client = MilvusClient(uri=milvus_instance_uri, token=milvus_token)
            return client
        except exceptions.MilvusException as e:
            raise MilvusHostError(milvus_instance_uri)

    def create_database(self, name):
        client = self.get_milvus_client(self.milvus_instance_uri,
                                        self.milvus_token)
        client.create_database(db_name=name)
        client.close()

    def create_schema(self, client):
        schema = client.create_schema(enable_dynamic_field=True)
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=self.embedding_dimention)
        index_params = client.prepare_index_params()
        index_params.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="L2")
        return schema, index_params
        
    def create_collection_milvus_db(self, client, collection_name="demo_collection"):
        if client.has_collection(collection_name) == False:
            schema, index_params = self.create_schema(client)
            client.create_collection(collection_name=collection_name,
                dimension=self.embedding_dimention,
                schema=schema, index_params=index_params)
            self.logger.debug(f"Collection '{collection_name}' has been created!")
        else:
            self.logger.debug(f"Collection '{collection_name}' has not been created!")

    def insert_docs_milvus_db(self, docs, collection_name="demo_collection"):
        client = self.get_milvus_client(self.milvus_instance_uri, self.milvus_token)
        vectors = self.embedding_fn.encode_documents(docs)
        data = [{"vector": vectors[i], "text": docs[i]} for i in range(len(vectors))]
        self.create_collection_milvus_db(client, collection_name)
        res = client.insert(collection_name=collection_name, data=data)
        client.close()
        self.logger.debug(res) 
                
    def search_milvus_db(self, query_text_arr, collection="demo_collection", result_limit=2, output_fields=["text", "subject"]):
        try:
            client = self.get_milvus_client(self.milvus_instance_uri, self.milvus_token)
            query_vectors = self.embedding_fn.encode_queries(query_text_arr)

            res = client.search(
                collection_name=collection,  # target collection
                data=query_vectors,  # query vectors
                limit=result_limit,  # number of returned entities
                output_fields=output_fields,  # specifies fields to be returned
            )

            result_text = "\n".join([ans['entity']['text'] for ans in res[0]])
            client.close()
            return result_text
        except Exception as e:
            self.logger.critical(e)

if __name__=="__main__":
    milvus_util = MilvusUtil()
    milvus_util.logger.info("Start")
    """ milvus_util.insert_docs_milvus_db(["During World War II, Turing worked for the Government Code and Cypher School at Bletchley Park, Britain's codebreaking centre that produced Ultra intelligence.",
                           "Artificial intelligence was founded as an academic discipline in 1956.",
    "Alan Turing was the first person to conduct substantial research in AI.",
    "Born in Maida Vale, London, Turing was raised in southern England."], "new_collection_1" )
    milvus_util.insert_docs_milvus_db(["Alan Turing was a good human being."], "new_collection_1" )  """
    
    milvus_util.logger.info(milvus_util.search_milvus_db(["who is Alan Turing?"],"new_collection_1", 3))