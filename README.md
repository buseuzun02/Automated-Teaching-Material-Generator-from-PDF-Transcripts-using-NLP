# Automated Teaching Material Generator from PDF Transcripts using NLP

This project leverages **Natural Language Processing (NLP)** techniques to transform PDF transcripts into structured teaching materials. It combines text summarization, keyword extraction, and contextual example generation to provide comprehensive educational content.

## Key Features
- **PDF Parsing:** Extract text content from uploaded PDF transcripts.  
- **Text Summarization:** Uses a BERT-based model (facebook/bart-large-cnn) to summarize long texts in manageable chunks.  
- **Keyword Extraction:** Identifies significant terms while filtering out common and irrelevant words.  
- **Example Generation:** Creates illustrative examples based on extracted keywords to enhance learning.  
- **Structured Material:** Produces sections like Introduction, Details, Examples, Expansion, and Conclusion for a complete teaching resource.  
- **Interactive Interface:** Built with **Gradio**, allowing easy upload of PDFs and retrieval of structured educational content.  
- **Scalable Output:** Generates detailed material, expandable to meet a minimum word count of 3900 words for comprehensive coverage.

## Purpose
Designed for educators and content creators, this tool **automates the transformation of transcripts into high-quality teaching materials**, saving time while maintaining pedagogical value. It demonstrates the practical integration of NLP models into real-world educational workflows.
