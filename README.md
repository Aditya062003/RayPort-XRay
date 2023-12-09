# CS550-project: Automated Radiology Report Generation and Disease Classification for enhanced patient care

This is the repository containing our CS550 project. The team members are:
- Aditya Sankhla (12140060)
- Aditya Vinay Dubey (12140100)
- Tushar Bansal (12141680)

## Objective
The objective of this project is to create a portal which will take in the patients (client) X-Ray along with a list of symptoms the client is suffering from through a chat bot and output a detailed report which will highlight the possible diseases that the patient might have based on an indepth analysis of the X-Ray along with the patient symptoms.

## Mid-Evaluation Progress
- Performed in-depth EDA and visualisation of the dataset and trained several CNN models on this dataset.
- Successfully trained a multimodal (14 classes) image classification model with high accuracy and recall.
- Developed a prototype portal (GUI) using tkinter.
- Implemented a simple hugging face NLP model to classify the diseases based on the user's text and the context provided.

## Final Evaluation Progress
   - Developed a well-crafted data pipeline, ensuring effective preprocessing and augmentation to enhance model performance.
   - Rigorously validated models, with DenseNet emerging as the optimal choice, showcasing superior accuracy and AUC values.
   - Integrated RAG with the Langchain Package to process user-reported symptoms through the chatbot.
   - Leveraged a retrieval model, document database, context embeddings, and a generative model for coherent and contextually relevant responses.
   - Implemented a React-based frontend for a seamless user interface.
   - Utilized Flask to manage requests and TensorFlow, Pinecone, and Langchain for core support.
   - Incorporated practical tools such as Base64 for file handling, ReportLab, WeasyPrint, and Flask-CORS for a seamless diagnostic and reporting experience.

### Overall Methodology:

1. **Robust Foundation for Medical Diagnostic Systems:**
   - Established a robust foundation through effective data preprocessing, model training, and advanced techniques like RAG.
   - Combined efforts resulted in a comprehensive diagnostic solution, integrating image processing and language understanding.

2. **Performance Metrics:**
   - Generated critical performance metrics (accuracy, AUC, precision, recall, F1-score, support) for all classes within the dataset, providing valuable insights into model effectiveness.

### Results

<img src="https://github.com/adismort14/CS550-project/assets/102402625/4947fd32-d6a9-401b-afd3-a4364899fc49" width="400"/> <br/>
<img src="https://github.com/adismort14/CS550-project/assets/102402625/c0b1473d-7b7d-4d68-8c43-87c8ca4cd170" width="400"/><br/>
<img src="https://github.com/adismort14/CS550-project/assets/102402625/14612993-1321-4d6d-b9b4-4ae970f538de" width="400"/><br/>



Our project represents a significant advancement in medical diagnostics, providing a user-friendly portal with intelligent diagnostic capabilities, thereby enhancing patient care through technology.
