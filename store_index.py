from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings
from langchain_chroma import Chroma
import os

extracted_data=load_pdf_file(data='Data/')
text_chunks=text_split(extracted_data)
embeddings = download_hugging_face_embeddings()

# Embed each chunk and upsert the embeddings into your Chroma index locally in batches.
persist_directory = "chroma_db"
batch_size = 5000

# Create the vector store with the first batch
docsearch = Chroma.from_documents(
    documents=text_chunks[:batch_size],
    embedding=embeddings,
    persist_directory=persist_directory
)

# Add the remaining batches
for i in range(batch_size, len(text_chunks), batch_size):
    batch = text_chunks[i:i + batch_size]
    docsearch.add_documents(documents=batch)

print("Index created successfully.")
