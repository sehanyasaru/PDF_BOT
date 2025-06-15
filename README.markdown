# PDF Question Answering System 🚀

## Overview
This project implements a question-answering system for PDF documents using large language models (LLMs) and vector embeddings. It processes multiple PDF files, extracts their text, creates embeddings using Google Generative AI, and enables question answering with source attribution using a Retrieval-Augmented Generation (RAG) approach. ✅

## Features
- **PDF Processing**: Extracts text from PDF files using `PyMuPDFLoader`. 📄
- **Text Chunking**: Splits documents into manageable chunks with `RecursiveCharacterTextSplitter`. ✂️
- **Embeddings**: Generates embeddings with `GoogleGenerativeAIEmbeddings` and stores them in a `FAISS` vector store. 🗄️
- **Question Answering**: Uses `RetrievalQAWithSourcesChain` with `ChatGoogleGenerativeAI` to answer queries with source references. ❓
- **Scalable**: Processes multiple PDFs and supports queries on their content. 📚

## Prerequisites
- Python 3.11.9 🐍
- Required libraries:
  ```bash
  pip install sentence-transformers langchain pymupdf faiss-cpu google-generativeai langchain-google-genai joblib
  ```
- Google API Key for `GoogleGenerativeAIEmbeddings` and `ChatGoogleGenerativeAI`. 🔑

## Setup
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/pdf-qa-system.git
   cd pdf-qa-system
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Environment Variable**:
   Add your Google API key to the environment:
   ```bash
   export GOOGLE_API_KEY='your-api-key'
   ```
   Or, in the code, set:
   ```python
   os.environ["GOOGLE_API_KEY"] = "your-api-key"
   ```

4. **Prepare PDF Files**:
   Place your PDF files in a directory (e.g., `ML(CS4052)`) and update the `pdf_paths` list in the code with their paths. 📂

## Usage
1. **Run the Jupyter Notebook**:
   - Open `PDF_LLM.ipynb` in Jupyter Notebook or JupyterLab. 📓
   - Execute the cells sequentially to:
     - Load and process PDFs.
     - Split text into chunks.
     - Create and save embeddings to a FAISS index.
     - Initialize the QA chain.
     - Query the system (e.g., "What is meant by binning?").

2. **Example Query**:
   ```python
   query = "What is meant by binning?"
   result = chain({"question": query}, return_only_outputs=True)
   print(result)
   ```
   **Output**:
   ```python
   {
       'answer': 'Binning is grouping continuous values into bins. For example, raw data of ages 22, 25, 37, 40, 55 can be binned into age groups "Young", "Middle-aged", and "Old".',
       'sources': 'C:\\Users\\User\\Desktop\\ML(CS4052)\\Feature Engineering.pdf'
   }
   ```

## Project Structure
- `PDF_LLM.ipynb`: Main Jupyter Notebook with the implementation. 📓
- `faiss_index/`: Directory storing the FAISS vector store. 🗄️
- `ML(CS4052)/`: Directory containing sample PDF files (e.g., `Feature Engineering.pdf`, `Neural Networks.pdf`, `Supervised Learning.pdf`). 📚

## Dependencies
- `sentence-transformers`: For sentence embeddings. 🧠
- `langchain`: For text splitting, document loading, and QA chain. 🔗
- `pymupdf`: For PDF text extraction. 📄
- `faiss-cpu`: For vector storage and retrieval. 🗃️
- `google-generativeai`: For Google Generative AI embeddings and LLM. 🌐
- `langchain-google-genai`: For integration with Google Generative AI. 🔌
- `joblib`: For serialization. 💾

## Notes
- Ensure the `GOOGLE_API_KEY` is valid and has access to the necessary Google APIs. 🔐
- The FAISS index is saved locally in the `faiss_index` directory for reuse. 📂
- The system is designed for CPU-based FAISS; for GPU support, install `faiss-gpu`. ⚡
- Update `pdf_paths` with the correct paths to your PDF files. 📍

## Contributing
1. Fork the repository. 🍴
2. Create a new branch (`git checkout -b feature-branch`). 🌱
3. Make your changes and commit (`git commit -m "Add feature"`). ✍️
4. Push to the branch (`git push origin feature-branch`). 🚀
5. Create a pull request. 🤝

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details. 📜

## Acknowledgments
- Built with [LangChain](https://github.com/langchain-ai/langchain) and [Google Generative AI](https://cloud.google.com/ai/generative-ai). 🙌
- Inspired by the need for efficient PDF-based question answering in educational contexts. 🎓