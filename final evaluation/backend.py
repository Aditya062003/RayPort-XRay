from flask import Flask, render_template, jsonify, request, send_file
import tensorflow as tf
import io
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import pinecone
from dotenv import load_dotenv
import os
from reportlab.pdfgen import canvas
from weasyprint import HTML

app = Flask(__name__)
load_dotenv()

model = tf.keras.models.load_model('model_DenseNet121_Full_Sample.h5')
loader = PyPDFLoader("disease_compendium.pdf")
data = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    length_function=len,
    add_start_index=True,
)
texts = text_splitter.split_documents(data)


embeddings = OpenAIEmbeddings(openai_api_key=os.getenv('OPENAI_API_KEY'))


pinecone.init(api_key='9f6644e9-2ab1-46a5-8d35-d5ade0ee39bf', environment='gcp-starter')
index_name = pinecone.Index('lung-disease')
vectordb = Pinecone.from_documents(texts, embeddings, index_name='lung-disease')
retriever = vectordb.as_retriever()


llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0)


memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
chain = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory)

def process_user_input(user_input):
    query ='The array provided symbolizes if the user has potentially a chest selected medically condition ' + user_input
    result = chain.run({'question': query})
    return result


def generate_pdf(result, filename):
    buffer = io.BytesIO()
    html_content = f"<html><body>{result}</body></html>"
    HTML(string=html_content).write_pdf(buffer)

    buffer.seek(0)
    with open(filename, 'wb') as f:
        f.write(buffer.read())

def preprocess_image(image):
    image = tf.image.grayscale_to_rgb(image)
    image = tf.image.resize(image, [224, 224])
    image_array = tf.image.convert_image_dtype(image, dtype=tf.uint8)
    image_array = tf.image.convert_image_dtype(image, dtype=tf.uint8)
    return image_array
    

def predict_label(image_data):
    image = tf.image.decode_jpeg(image_data)
    preprocessed_image = preprocess_image(image)

    prediction = model.predict(tf.expand_dims(preprocessed_image, axis=0))[0]

    prediction_list = prediction.tolist()  

    return prediction_list

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    image_data = file.read()

    p = predict_label(image_data)

    print("Predictions:", p)
    result = process_user_input(str(p))
    print(result)

    pdf_filename = 'output.pdf'
    generate_pdf(result, pdf_filename)

    return send_file(pdf_filename, as_attachment=True)

    return jsonify({
        'prediction': result
    })


if __name__ == '__main__':
    app.run(debug=True)
