from google import genai
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv('GENAI_API_KEY'))

MODEL_ID = "gemini-embedding-001"

key = "I'm not sure about, but I think I'm going to just send it."
result = client.models.embed_content(model=MODEL_ID, contents=[key])
[embedding] = result.embeddings
input_text_embedding = np.array(embedding.values).reshape(1, -1)


text = [
    "Boosting Product Quality",
    "Increasing Market Share",
    "Implementing Sustainable Manufacturing",
    "Upholding Regulatory Compliance",
    "Promoting Employee Engagement",
    "Fostering Innovation",
    "Strengthening Supplier Relations",
    "Reducing Production Costs",
    "Improving Knowledge Management",
    "Enhancing Customer Satisfaction"
]

df = pd.DataFrame(text, columns=["text"])

df["embeddings"] = df.apply(lambda x: client.models.embed_content(model=MODEL_ID, contents=(x['text'])).embeddings[0].values, axis=1)
print(df)




for index, row in df.iterrows():
    okr = np.array(row['embeddings']).reshape(1, -1)
    val = cosine_similarity(okr, input_text_embedding)
    print(f"Similarity between \"{row['text']}\" and \"{key}\" = {val[0][0]}")

