try:
  with open("jne.txt", "r") as notes:
    getnotes = []
    for lines in notes:
      getnotes.append(lines.strip("\r\n"))
    notes.close()
    path = getnotes[0]
    path = '/'.join(path.split('\\'))
except:
  path = ""
  try: import ctypes; ctypes.windll.user32.MessageBoxW(0, "check path...", "Python", 1)
  except: print("check path...")

data_file = path + "data_source.txt"
control_data = path + "control_data.xlsx"

get_data = open(data_file, "r")
data_jne = get_data.readlines()

import pandas as pd
get_control_data = pd.read_excel(control_data) #133497
col_data = get_control_data["terms"].apply(str)
control_data_jne = [str(row) for row in col_data]

import chromadb
from chromadb.utils import embedding_functions
from chromadb import Documents, EmbeddingFunction, Embeddings
client = chromadb.Client()
#dbPath = path + "chroma"
#client = chromadb.PersistentClient(path=dbPath)

import torch
from transformers import AutoTokenizer, AutoModel
#model = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-de', trust_remote_code=True, torch_dtype=torch.bfloat16)
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/LaBSE')
#model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
#model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

class Embedding_Function(EmbeddingFunction):
  def __call__(self, input: Documents) -> Embeddings:
    embeddings = model.encode(input)
    return(embeddings.tolist())

my_embeddings = Embedding_Function()


dbName = "checkStr"
dbDocs = data_jne

collection = client.get_or_create_collection(
#collection = client.create_collection(
  name=dbName,
  embedding_function=my_embeddings,
  metadata={"hnsw:space": "cosine"}
)

for i in range(len(dbDocs)):
  collection.add(
    documents=dbDocs[i],
    ids=str(i),
    metadatas={"source": "jne_data"}
  )

dist_threshold = 0.15
results = []

for i in range(len(control_data_jne)):
  db_query = collection.query(query_texts=[control_data_jne[i]], n_results=len(dbDocs))
  nearest_embeddings = db_query['ids'][0]
  embedding_document = db_query['documents'][0]
  distances = db_query['distances'][0]
  #filtered_results = [(id, doc, dist) for id, doc, dist in zip(nearest_embeddings, embedding_document, distances) if dist <= dist_threshold]
  filtered_results = [(id, dist) for id, dist in zip(nearest_embeddings, distances) if dist <= dist_threshold]
  results.append(len(filtered_results))

#results = '\n'.join(str(x) for x in results)
print(results)
#with open('results.txt', 'w') as output_file:
#  output_file.writelines(str(val) + "\n" for val in results)

write_result = get_control_data.assign(PyVals=results)
result_file = path + "result.xlsx"
write_result.to_excel(result_file, index=False)


