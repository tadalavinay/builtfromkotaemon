"""
Document loader and processor for the chatbot application
"""

import os
from pathlib import Path
from typing import List, Dict, Any
import json
import pandas as pd
from datetime import datetime

class DocumentLoader:
    """Load and process documents for the chatbot"""
    
    def __init__(self, documents_dir: Path):
        self.documents_dir = Path(documents_dir)
        self.documents_dir.mkdir(exist_ok=True)
    
    def create_sample_documents(self) -> List[Dict[str, Any]]:
        """Create sample documents for testing"""
        
        sample_docs = [
            {
                "id": "doc_001",
                "title": "Introduction to Machine Learning",
                "content": """Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It focuses on developing computer programs that can access data and use it to learn for themselves.

The process of learning begins with observations or data, such as examples, direct experience, or instruction, in order to look for patterns in data and make better decisions in the future based on the examples that we provide. The primary aim is to allow the computers to learn automatically without human intervention or assistance and adjust actions accordingly.

Machine learning algorithms are often categorized as supervised or unsupervised. Supervised algorithms require a data scientist or data analyst with machine learning skills to provide both input and desired output, in addition to furnishing feedback about the accuracy of predictions during algorithm training. Data scientists determine which variables, or features, the model should analyze and use to develop predictions.

Unsupervised algorithms do not need to be trained with desired outcome data. Instead, they use an iterative approach called deep learning to review data and arrive at conclusions. Unsupervised learning algorithms are used for more complex processing tasks than supervised learning systems.""",
                "metadata": {
                    "category": "technology",
                    "tags": ["machine learning", "AI", "algorithms"],
                    "author": "AI Assistant",
                    "created_date": datetime.now().isoformat()
                }
            },
            {
                "id": "doc_002",
                "title": "Python Programming Basics",
                "content": """Python is a high-level, interpreted programming language known for its simplicity and readability. Created by Guido van Rossum and first released in 1991, Python's design philosophy emphasizes code readability with its notable use of significant whitespace.

Python supports multiple programming paradigms, including structured, object-oriented, and functional programming. It is often described as a "batteries included" language due to its comprehensive standard library.

Key features of Python include:
- Easy to learn and use
- Interpreted and interactive
- Object-oriented
- Extensive standard library
- Portable across operating systems
- Free and open source

Python is widely used in various domains including web development, data science, artificial intelligence, scientific computing, and automation. Popular frameworks and libraries include Django for web development, NumPy and Pandas for data analysis, TensorFlow and PyTorch for machine learning, and many more.""",
                "metadata": {
                    "category": "programming",
                    "tags": ["python", "programming", "basics"],
                    "author": "AI Assistant",
                    "created_date": datetime.now().isoformat()
                }
            },
            {
                "id": "doc_003",
                "title": "Data Science Fundamentals",
                "content": """Data science is an interdisciplinary field that uses scientific methods, processes, algorithms, and systems to extract knowledge and insights from structured and unstructured data. It combines domain expertise, programming skills, and knowledge of mathematics and statistics to extract meaningful insights from data.

The data science lifecycle typically involves:
1. Data collection and acquisition
2. Data cleaning and preprocessing
3. Exploratory data analysis
4. Feature engineering
5. Model building and training
6. Model evaluation and validation
7. Deployment and monitoring

Key tools and technologies in data science include:
- Programming languages: Python, R, SQL
- Data manipulation: Pandas, NumPy
- Visualization: Matplotlib, Seaborn, Plotly
- Machine learning: Scikit-learn, TensorFlow, PyTorch
- Big data: Apache Spark, Hadoop
- Cloud platforms: AWS, Google Cloud, Azure

Data scientists work on various problems including predictive analytics, recommendation systems, natural language processing, computer vision, and business intelligence.""",
                "metadata": {
                    "category": "data science",
                    "tags": ["data science", "analytics", "machine learning"],
                    "author": "AI Assistant",
                    "created_date": datetime.now().isoformat()
                }
            }
        ]
        
        # Save documents to files
        for doc in sample_docs:
            filename = f"{doc['id']}_{doc['title'].lower().replace(' ', '_')}.json"
            filepath = self.documents_dir / filename
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(doc, f, indent=2, ensure_ascii=False)
        
        return sample_docs
    
    def load_documents(self) -> List[Dict[str, Any]]:
        """Load all documents from the documents directory"""
        documents = []
        
        for file_path in self.documents_dir.glob("*.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    doc = json.load(f)
                    documents.append(doc)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        return documents
    
    def load_documents_as_dataframe(self) -> pd.DataFrame:
        """Load documents as a pandas DataFrame for processing"""
        documents = self.load_documents()
        
        if not documents:
            return pd.DataFrame()
        
        # Flatten metadata into separate columns
        df_data = []
        for doc in documents:
            row = {
                'id': doc['id'],
                'title': doc['title'],
                'content': doc['content'],
            }
            # Add metadata fields
            if 'metadata' in doc:
                row.update(doc['metadata'])
            df_data.append(row)
        
        return pd.DataFrame(df_data)
    
    def add_document(self, title: str, content: str, metadata: Dict[str, Any] = None) -> str:
        """Add a new document to the collection"""
        doc_id = f"doc_{int(datetime.now().timestamp())}"
        
        document = {
            "id": doc_id,
            "title": title,
            "content": content,
            "metadata": metadata or {}
        }
        
        filename = f"{doc_id}_{title.lower().replace(' ', '_')}.json"
        filepath = self.documents_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(document, f, indent=2, ensure_ascii=False)
        
        return doc_id
    
    def get_document_count(self) -> int:
        """Get the number of documents in the collection"""
        return len(list(self.documents_dir.glob("*.json")))
    
    def list_documents(self) -> List[Dict[str, str]]:
        """List all documents with basic info"""
        documents = []
        
        for file_path in self.documents_dir.glob("*.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    doc = json.load(f)
                    documents.append({
                        'id': doc['id'],
                        'title': doc['title'],
                        'category': doc.get('metadata', {}).get('category', 'unknown')
                    })
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
        
        return documents

if __name__ == "__main__":
    # Test the document loader
    from config import DOCUMENTS_DIR
    
    loader = DocumentLoader(DOCUMENTS_DIR)
    
    # Create sample documents
    print("Creating sample documents...")
    docs = loader.create_sample_documents()
    print(f"Created {len(docs)} sample documents")
    
    # List documents
    print("\nDocuments in collection:")
    for doc in loader.list_documents():
        print(f"- {doc['title']} ({doc['category']})")
    
    print(f"\nTotal documents: {loader.get_document_count()}")