import os
import json
import asyncio
import pickle
from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI

API_KEY = ""
BASE_URL = ""

llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini", openai_api_key=API_KEY, openai_api_base=BASE_URL)
llm_transformer = LLMGraphTransformer(llm=llm)

async def main():
    with open('AgentJudge-strict-raw.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    documents = [Document(page_content=item['raw_record'], metadata={**item}) for item in data if item['ambiguous'] == 0]

    batch_size = 10
    graph_documents = []
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        graph_documents_batch = await llm_transformer.aconvert_to_graph_documents(batch)
        if i == 0:
            graph_documents = graph_documents_batch
        else:
            graph_documents.extend(graph_documents_batch)

    output_filename = 'graph_documents.pkl'
    with open(output_filename, 'wb') as f: 
        pickle.dump(graph_documents, f)

if __name__ == "__main__":
    asyncio.run(main())