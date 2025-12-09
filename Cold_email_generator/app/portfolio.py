# portfolio.py

import pandas as pd
import chromadb
import uuid
import os


class Portfolio:
    def __init__(self, file_path="app/resource/my_portfolio.csv", collection_name="portfolio"):
        """
        Initialize the Portfolio class.
        Loads CSV file and sets up Chroma persistent client and collection.
        """
        self.file_path = file_path
        self.data = pd.read_csv(file_path)

        # Disable telemetry
        os.environ["CHROMA_TELEMETRY"] = "FALSE"

        # Initialize Chroma client
        self.chroma_client = chromadb.PersistentClient(path='vectorstore')
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name)

    def load_portfolio(self):
        """
        Load portfolio data into Chroma collection if empty.
        Each Techstack is added as a document with a unique UUID.
        """
        if self.collection.count() == 0:
            for _, row in self.data.iterrows():
                self.collection.add(
                    documents=[str(row["Techstack"])],      # must be list
                    metadatas=[{"links": row["Links"]}],   # must be list
                    ids=[str(uuid.uuid4())]                 # must be list
                )
        print("Portfolio data loaded successfully!")

    def query_links(self, skills):
        """
        Query portfolio collection for relevant links based on skills.
        Returns a flattened list of links.
        """
        result = self.collection.query(
            query_texts=skills, n_results=5).get('metadatas', [])

        # Flatten the result safely
        links = []
        if result:
            for sublist in result:
                if sublist:  # check sublist is not None
                    for item in sublist:
                        if 'links' in item and item['links']:
                            links.append(item['links'])
        return links
