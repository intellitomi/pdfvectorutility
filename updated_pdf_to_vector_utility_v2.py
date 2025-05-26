"""
PDF to Vector Database Utility

This module provides functionality to:
1. Extract text from PDFs (via direct extraction or Grobid)
2. Clean and preprocess the extracted text
3. Generate embeddings using transformers
4. Store embeddings in Pinecone vector database
"""

import os
import re
import json
import requests
import tempfile
import logging
from typing import Dict, List, Tuple, Union, Optional, Any
from pathlib import Path

import fitz  # PyMuPDF
from pinecone import Pinecone, ServerlessSpec  # Updated Pinecone import
import numpy as np
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import torch

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PDFProcessor:
    """Class to handle PDF text extraction with support for both direct extraction and Grobid."""
    
    def __init__(self, grobid_url: Optional[str] = None):
        """
        Initialize the PDF processor.
        
        Args:
            grobid_url: URL to the Grobid service (e.g., "http://localhost:8070"). 
                        If None, will use direct extraction.
        """
        self.grobid_url = grobid_url
        logger.info(f"PDF Processor initialized {'with Grobid support' if grobid_url else 'with direct extraction'}")
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from a PDF file using PyMuPDF.
        
        Args:
            pdf_path: Path to the PDF file.
            
        Returns:
            Extracted text as a string.
        """
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            logger.info(f"Successfully extracted text from {pdf_path} using PyMuPDF")
            return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            raise
    
    def process_with_grobid(self, pdf_path: str) -> str:
        """
        Process a PDF file with Grobid to extract structured text.
        
        Args:
            pdf_path: Path to the PDF file.
            
        Returns:
            TEI XML as a string.
        """
        if not self.grobid_url:
            raise ValueError("Grobid URL is not configured")
        
        try:
            # Prepare the PDF file for upload
            with open(pdf_path, 'rb') as pdf_file:
                files = {'input': (os.path.basename(pdf_path), pdf_file, 'application/pdf')}
                
                # Make request to Grobid's processFulltextDocument endpoint
                url = f"{self.grobid_url}/api/processFulltextDocument"
                response = requests.post(url, files=files)
                
                if response.status_code == 200:
                    logger.info(f"Successfully processed {pdf_path} with Grobid")
                    return response.text
                else:
                    logger.error(f"Grobid processing failed: {response.status_code} - {response.text}")
                    raise Exception(f"Grobid processing failed: {response.status_code}")
                    
        except Exception as e:
            logger.error(f"Error processing PDF with Grobid: {e}")
            raise
    
    def parse_grobid_xml(self, xml_content: str) -> Dict[str, Any]:
        """
        Parse the TEI XML output from Grobid into a structured dictionary.
        
        Args:
            xml_content: TEI XML string from Grobid.
            
        Returns:
            Dictionary containing structured document information.
        """
        try:
            soup = BeautifulSoup(xml_content, 'xml')
            
            # Extract basic metadata
            result = {
                'title': '',
                'abstract': '',
                'authors': [],
                'body_text': [],
                'references': []
            }
            
            # Title
            title_tag = soup.find('titleStmt')
            if title_tag and title_tag.find('title'):
                result['title'] = title_tag.find('title').text.strip()
            
            # Abstract
            abstract_tag = soup.find('abstract')
            if abstract_tag:
                result['abstract'] = abstract_tag.get_text(' ', strip=True)
            
            # Authors
            author_tags = soup.find_all('author')
            for author_tag in author_tags:
                persname = author_tag.find('persName')
                if persname:
                    forename = persname.find('forename')
                    surname = persname.find('surname')
                    name_parts = []
                    if forename:
                        name_parts.append(forename.text.strip())
                    if surname:
                        name_parts.append(surname.text.strip())
                    if name_parts:
                        result['authors'].append(' '.join(name_parts))
            
            # Body text by section
            div_tags = soup.find_all('div')
            for div in div_tags:
                head = div.find('head')
                section_title = head.text.strip() if head else "Unnamed Section"
                
                paragraphs = div.find_all('p')
                section_text = ' '.join([p.text.strip() for p in paragraphs])
                
                if section_text:
                    result['body_text'].append({
                        'section': section_title,
                        'text': section_text
                    })
            
            # References/Bibliography
            bib_tags = soup.find_all('biblStruct')
            for bib in bib_tags:
                ref = {}
                
                # Title
                title = bib.find('title', type='main')
                if title:
                    ref['title'] = title.text.strip()
                
                # Authors
                ref_authors = []
                author_tags = bib.find_all('author')
                for author_tag in author_tags:
                    persname = author_tag.find('persName')
                    if persname:
                        forename = persname.find('forename')
                        surname = persname.find('surname')
                        name_parts = []
                        if forename:
                            name_parts.append(forename.text.strip())
                        if surname:
                            name_parts.append(surname.text.strip())
                        if name_parts:
                            ref_authors.append(' '.join(name_parts))
                
                if ref_authors:
                    ref['authors'] = ref_authors
                    
                # Year
                date = bib.find('date')
                if date and date.get('when'):
                    ref['year'] = date.get('when')[:4]  # Extract year from date
                
                if ref:
                    result['references'].append(ref)
            
            logger.info("Successfully parsed Grobid XML output")
            return result
            
        except Exception as e:
            logger.error(f"Error parsing Grobid XML: {e}")
            raise
    
    def parse_tei_xml_file(self, xml_path: str) -> Dict[str, Any]:
        """
        Parse a TEI XML file into a structured dictionary.
        
        Args:
            xml_path: Path to the TEI XML file.
            
        Returns:
            Dictionary containing structured document information.
        """
        try:
            with open(xml_path, 'r', encoding='utf-8') as f:
                xml_content = f.read()
            return self.parse_grobid_xml(xml_content)
        except Exception as e:
            logger.error(f"Error parsing TEI XML file: {e}")
            raise

class TextProcessor:
    """Class to clean and preprocess text."""
    
    def __init__(self):
        """Initialize the text processor with default cleaning rules."""
        pass
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Input text to clean.
            
        Returns:
            Cleaned text.
        """
        # Replace multiple whitespace with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters except punctuation
        text = re.sub(r'[^\w\s\.,;:!?\'\"()-]', '', text)
        
        # Normalize whitespace
        text = text.strip()
        
        return text
    
    def segment_text(self, text: str, max_length: int = 1000, overlap: int = 100) -> List[str]:
        """
        Segment text into chunks of specified maximum length with overlap,
        with improved memory efficiency.
    
        Args:
            text: Text to segment.
            max_length: Maximum length of each chunk.
            overlap: Number of characters to overlap between chunks.
            
        Returns:
            List of text segments.
        """
        if len(text) <= max_length:
            return [text]
        
        segments = []
        start = 0
        
        while start < len(text):
            # Find a good breakpoint (period, question mark, exclamation)
            end = min(start + max_length, len(text))
            
            if end < len(text):
                # Look for sentence boundaries only within a smaller window
                # This avoids creating large slices of the text in memory
                search_end = max(start, end - 200)  # Only look in the last 200 chars
                search_text = text[search_end:end]
                
                # Find the last sentence boundary in the search window
                last_period = search_text.rfind('. ')
                last_question = search_text.rfind('? ')
                last_exclamation = search_text.rfind('! ')
                
                # Get the position of the latest boundary
                breakpoint_relative = max(last_period, last_question, last_exclamation)
                
                if breakpoint_relative > -1:  # If found within the search window
                    breakpoint = search_end + breakpoint_relative + 1  # +1 to include the punctuation
                    end = breakpoint
            
            # Extract only the required segment
            segment = text[start:end].strip()
            segments.append(segment)
            
            # Move forward
            start = max(start, end - overlap)
        
        return segments
    
    def process_document(self, doc_data: Dict[str, Any], max_length: int = 1000, overlap: int = 100) -> List[Dict[str, str]]:
        """
        Process a structured document dictionary into segments ready for embedding.
        
        Args:
            doc_data: Document dictionary from PDF processor.
            max_length: Maximum length of each segment.
            overlap: Overlap between segments.
            
        Returns:
            List of dictionaries containing segment metadata and text.
        """
        segments = []
        
        # Process title and abstract as one segment
        header_text = f"Title: {doc_data.get('title', '')}\n"
        if doc_data.get('abstract'):
            header_text += f"Abstract: {doc_data.get('abstract')}\n"
        
        if header_text.strip():
            header_text = self.clean_text(header_text)
            segments.append({
                'segment_type': 'header',
                'section': 'header',
                'text': header_text
            })
        
        # Process body text
        for section in doc_data.get('body_text', []):
            section_title = section.get('section', 'Unknown Section')
            section_text = section.get('text', '')
            
            if not section_text:
                continue
                
            cleaned_text = self.clean_text(section_text)
            text_segments = self.segment_text(cleaned_text, max_length, overlap)
            
            for i, segment in enumerate(text_segments):
                segments.append({
                    'segment_type': 'body',
                    'section': section_title,
                    'segment_index': i,
                    'text': segment
                })
        
        return segments

class TextEmbedder:
    """Class to generate embeddings for text using transformer models."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the text embedder with a specific model.
        
        Args:
            model_name: Name of the transformer model to use for embeddings.
        """
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        logger.info(f"Text embedder initialized with model {model_name} on {self.device}")
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a text segment.
        
        Args:
            text: Text to embed.
            
        Returns:
            Numpy array containing the embedding vector.
        """
        # Mean Pooling function
        def mean_pooling(model_output, attention_mask):
            token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        # Tokenize and prepare for the model
        encoded_input = self.tokenizer(
            text, 
            padding=True, 
            truncation=True, 
            max_length=512, 
            return_tensors='pt'
        ).to(self.device)
        
        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        
        # Perform pooling
        sentence_embedding = mean_pooling(model_output, encoded_input['attention_mask'])
        
        # Normalize embeddings
        sentence_embedding = torch.nn.functional.normalize(sentence_embedding, p=2, dim=1)
        
        # Return as numpy array
        return sentence_embedding.cpu().numpy()[0]
    
    def batch_generate_embeddings(self, texts: List[str], batch_size: int = 8) -> List[np.ndarray]:
        """
        Generate embeddings for a list of texts in batches.
        
        Args:
            texts: List of texts to embed.
            batch_size: Batch size for processing.
            
        Returns:
            List of embedding vectors.
        """
        embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
            batch_texts = texts[i:i+batch_size]
            batch_embeddings = [self.generate_embedding(text) for text in batch_texts]
            embeddings.extend(batch_embeddings)
        
        return embeddings

class VectorDatabaseManager:
    """Class to handle interactions with Pinecone vector database."""
    
    def __init__(
        self, 
        api_key: str, 
        environment: str,
        index_name: str, 
        dimension: int = 384
    ):
        """
        Initialize the Pinecone database manager.
        
        Args:
            api_key: Pinecone API key.
            environment: Pinecone environment.
            index_name: Name of the Pinecone index to use.
            dimension: Dimension of the embedding vectors.
        """
        self.api_key = api_key
        self.environment = environment
        self.index_name = index_name
        self.dimension = dimension
        
        # Initialize Pinecone (updated for new API)
        self.pc = Pinecone(api_key=api_key)
        
        # Check if index exists, create if it doesn't
        existing_indexes = self.pc.list_indexes().names()
        if index_name not in existing_indexes:
            logger.info(f"Creating new Pinecone index: {index_name}")
            self.pc.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=environment.split('-')[2] if '-' in environment else 'aws',  # Extract cloud from environment
                    region=environment.split('-')[0] + '-' + environment.split('-')[1]  # Extract region from environment
                )
            )
        
        # Connect to the index
        self.index = self.pc.Index(index_name)
        logger.info(f"Connected to Pinecone index: {index_name}")
    
    def upsert_vectors(
        self, 
        vectors: List[np.ndarray], 
        metadata_list: List[Dict[str, Any]], 
        id_prefix: str = ""
    ) -> None:
        """
        Upsert vectors to the Pinecone index.
        
        Args:
            vectors: List of embedding vectors.
            metadata_list: List of metadata dictionaries for each vector.
            id_prefix: Prefix for vector IDs.
        """
        items_to_upsert = []
        
        for i, (vector, metadata) in enumerate(zip(vectors, metadata_list)):
            vector_id = f"{id_prefix}_{i}" if id_prefix else str(i)
            
            items_to_upsert.append((
                vector_id,
                vector.tolist(),
                metadata
            ))
        
        # Upsert in batches to avoid payload size limitations
        batch_size = 100
        for i in range(0, len(items_to_upsert), batch_size):
            batch = items_to_upsert[i:i+batch_size]
            self.index.upsert(vectors=batch)
        
        logger.info(f"Upserted {len(vectors)} vectors to Pinecone index")
    
    def search_vectors(self, query_vector: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar vectors in the Pinecone index.
        
        Args:
            query_vector: Query embedding vector.
            top_k: Number of results to return.
            
        Returns:
            List of search results with metadata.
        """
        results = self.index.query(
            vector=query_vector.tolist(),
            top_k=top_k,
            include_metadata=True
        )
        
        return results.get('matches', [])
    
    def delete_vectors(self, ids: List[str]) -> None:
        """
        Delete vectors from the index.
        
        Args:
            ids: List of vector IDs to delete.
        """
        self.index.delete(ids=ids)
        logger.info(f"Deleted {len(ids)} vectors from Pinecone index")

class PDFToVectorPipeline:
    """Main class to orchestrate the PDF to vector pipeline."""
    
    def __init__(
        self,
        pinecone_api_key: str,
        pinecone_environment: str,
        pinecone_index_name: str,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        grobid_url: Optional[str] = None
    ):
        """
        Initialize the full pipeline.
        
        Args:
            pinecone_api_key: Pinecone API key.
            pinecone_environment: Pinecone environment.
            pinecone_index_name: Name of the Pinecone index.
            embedding_model: Name of the transformer model to use.
            grobid_url: URL to the Grobid service (optional).
        """
        self.pdf_processor = PDFProcessor(grobid_url=grobid_url)
        self.text_processor = TextProcessor()
        self.embedder = TextEmbedder(model_name=embedding_model)
        self.vector_db = VectorDatabaseManager(
            api_key=pinecone_api_key,
            environment=pinecone_environment,
            index_name=pinecone_index_name,
            dimension=self._get_embedding_dimension(embedding_model)
        )
        self.grobid_url = grobid_url
        logger.info("PDF to Vector pipeline initialized successfully")
    
    def _get_embedding_dimension(self, model_name: str) -> int:
        """
        Get the embedding dimension for a specific model.
        
        Args:
            model_name: Name of the transformer model.
            
        Returns:
            Embedding dimension.
        """
        if "MiniLM-L6" in model_name:
            return 384
        elif "mpnet" in model_name:
            return 768
        elif "bert-base" in model_name:
            return 768
        # Add more models as needed
        else:
            # Default for sentence-transformers models
            return 384
    
    def process_pdf_file(self, pdf_path: str, use_grobid: bool = False, doc_id: Optional[str] = None) -> str:
        """Modified to handle large files more gracefully"""
        logger.info(f"Processing PDF file: {pdf_path}")
        
        # Generate document ID if not provided
        if not doc_id:
            doc_id = os.path.basename(pdf_path).replace('.pdf', '')
        
        try:
            # Check file size first
            file_size_mb = os.path.getsize(pdf_path) / (1024 * 1024)
            
            # Use direct extraction for large files
            if file_size_mb > 10 and use_grobid:  # If file is larger than 10MB
                logger.warning(f"File {pdf_path} is {file_size_mb:.2f}MB - too large for Grobid, using direct extraction")
                use_grobid = False
                
            # Extract text using Grobid or direct extraction
            if use_grobid and self.grobid_url:
                xml_content = self.pdf_processor.process_with_grobid(pdf_path)
                doc_data = self.pdf_processor.parse_grobid_xml(xml_content)
            else:
                text = self.pdf_processor.extract_text_from_pdf(pdf_path)
                doc_data = {
                    'title': os.path.basename(pdf_path),
                    'abstract': '',
                    'body_text': [{'section': 'main', 'text': text}]
                }
            
            # Use the new method for large files
            segments = self.text_processor.process_large_document(doc_data, max_length=500, overlap=50)
        
        
            
            # Create metadata and extract text for embeddings
            texts = []
            metadata_list = []
            
            for i, segment in enumerate(segments):
                text = segment['text']
                
                # Create metadata for the segment
                metadata = {
                    'document_id': doc_id,
                    'segment_id': i,
                    'segment_type': segment.get('segment_type', 'unknown'),
                    'section': segment.get('section', 'unknown'),
                    'text_preview': text[:100] + '...' if len(text) > 100 else text
                }
                
                if 'title' in doc_data:
                    metadata['document_title'] = doc_data['title']
                
                texts.append(text)
                metadata_list.append(metadata)
            
            # Generate embeddings
            embeddings = self.embedder.batch_generate_embeddings(texts)
            
            # Store in vector database
            self.vector_db.upsert_vectors(
                vectors=embeddings,
                metadata_list=metadata_list,
                id_prefix=doc_id
            )
            
            logger.info(f"Successfully processed PDF: {pdf_path}")
            return doc_id
            
        except Exception as e:
            logger.error(f"Error processing PDF file {pdf_path}: {e}")
            raise
    
    def process_tei_xml_file(
        self, 
        xml_path: str,
        doc_id: Optional[str] = None
    ) -> str:
        """
        Process a TEI XML file through the pipeline.
        
        Args:
            xml_path: Path to TEI XML file.
            doc_id: Optional document ID for vector storage.
            
        Returns:
            Document ID used for vector storage.
        """
        logger.info(f"Processing TEI XML file: {xml_path}")
        
        # Generate document ID if not provided
        if not doc_id:
            doc_id = os.path.basename(xml_path).replace('.xml', '').replace('.tei', '')
        
        try:
            # Parse the XML file
            doc_data = self.pdf_processor.parse_tei_xml_file(xml_path)
            
            # Process and segment the document
            segments = self.text_processor.process_document(doc_data)
            
            # Create metadata and extract text for embeddings
            texts = []
            metadata_list = []
            
            for i, segment in enumerate(segments):
                text = segment['text']
                
                # Create metadata for the segment
                metadata = {
                    'document_id': doc_id,
                    'segment_id': i,
                    'segment_type': segment.get('segment_type', 'unknown'),
                    'section': segment.get('section', 'unknown'),
                    'text_preview': text[:100] + '...' if len(text) > 100 else text
                }
                
                if 'title' in doc_data:
                    metadata['document_title'] = doc_data['title']
                
                texts.append(text)
                metadata_list.append(metadata)
            
            # Generate embeddings
            embeddings = self.embedder.batch_generate_embeddings(texts)
            
            # Store in vector database
            self.vector_db.upsert_vectors(
                vectors=embeddings,
                metadata_list=metadata_list,
                id_prefix=doc_id
            )
            
            logger.info(f"Successfully processed TEI XML: {xml_path}")
            return doc_id
            
        except Exception as e:
            logger.error(f"Error processing TEI XML file {xml_path}: {e}")
            raise
    
    def search_documents(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for documents using a text query.
        
        Args:
            query_text: Text query.
            top_k: Number of results to return.
            
        Returns:
            List of search results with metadata.
        """
        # Generate embedding for the query
        query_embedding = self.embedder.generate_embedding(query_text)
        
        # Search the vector database
        results = self.vector_db.search_vectors(query_embedding, top_k=top_k)
        
        return results
    
    def batch_process_directory(
        self, 
        directory_path: str, 
        file_extension: str = 'pdf',
        use_grobid: bool = False
    ) -> List[str]:
        """
        Process all files with a specific extension in a directory.
        
        Args:
            directory_path: Path to the directory.
            file_extension: Extension of files to process ('pdf' or 'xml').
            use_grobid: Whether to use Grobid for processing PDFs.
            
        Returns:
            List of document IDs processed.
        """
        doc_ids = []
        directory = Path(directory_path)
        
        for file_path in directory.glob(f"*.{file_extension}"):
            try:
                if file_extension.lower() == 'pdf':
                    doc_id = self.process_pdf_file(str(file_path), use_grobid=use_grobid)
                elif file_extension.lower() in ['xml', 'tei']:
                    doc_id = self.process_tei_xml_file(str(file_path))
                else:
                    logger.warning(f"Unsupported file extension: {file_extension}")
                    continue
                    
                doc_ids.append(doc_id)
                
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")
        
        return doc_ids


# Example usage
if __name__ == "__main__":
    # Set up configuration
    config = {
        'pinecone_api_key': 'pcsk_5aPZ1y_6rjtPJnzsfMQrmkfG4KhoR3VL8djNzxzipydWPQZc5i6woBjbGDqBVUvbyweasJ',
        'pinecone_environment': 'us-east-1',  # Region format: 'region-cloud' (e.g., 'us-west-2-aws')
        'pinecone_index_name': 'pdfvector',
        'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
        'grobid_url': 'https://kermitt2-grobid.hf.space/'  # Optional, remove if not using Grobid
    }
    
    # Initialize the pipeline
    pipeline = PDFToVectorPipeline(
        pinecone_api_key=config['pinecone_api_key'],
        pinecone_environment=config['pinecone_environment'],
        pinecone_index_name=config['pinecone_index_name'],
        embedding_model=config['embedding_model'],
        grobid_url=config.get('grobid_url')  # Optional
    )
    
    # doc_id = pipeline.process_pdf_file('inputs/pdf/Soulonice.pdf', use_grobid=True)
    
    doc_id = pipeline.process_tei_xml_file('inputs/tei/Soulonice.pdf.tei.xml')

    # Example: Process a single PDF file
    # doc_id = pipeline.process_pdf_file('path/to/document.pdf', use_grobid=True)
    
    # Example: Process a TEI XML file
    # doc_id = pipeline.process_tei_xml_file('path/to/document.tei.xml')
    
    # Example: Batch process a directory
    # doc_ids = pipeline.batch_process_directory('path/to/pdf/directory', file_extension='pdf', use_grobid=True)
    
    # Example: Search for documents
    # results = pipeline.search_documents("quantum computing applications", top_k=5)
    # for i, result in enumerate(results):
    #     print(f"Result {i+1}:")
    #     print(f"Score: {result['score']}")
    #     print(f"Document: {result['metadata']['document_title']}")
    #     print(f"Section: {result['metadata']['section']}")
    #     print(f"Preview: {result['metadata']['text_preview']}")
    #     print("----")
