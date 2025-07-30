# Techno-Medical-Chatbot
AI-powered medical chatbot using LangChain, OpenAI, and Pinecone, deployed with Flask, Docker, and AWS CI/CD via GitHub Actions.


git clone https://github.com/Om-Bhati/Techno-Medical-Chatbot.git

conda create -n technomedibot python=3.10 -y
conda activate technomedibot

pip install -r requirements.txt

PINECONE_API_KEY = "************" # techno-medical-chatbot
GITHUB_TOKEN = "***********"


python store_index.py

python app.py

# TECH STACK USED 

Python
LangChain
Flask
GPT / git
Pinecone