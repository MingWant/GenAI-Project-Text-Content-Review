#pip install langchain langchain_openai langchain_community langchain-core flask flask-cors python-dotenv
import os
import json
from langchain_core.prompts import ChatPromptTemplate #寫Prompt用的
from langchain_core.output_parsers import StrOutputParser #整理格式用的
from langchain_core.runnables import RunnablePassthrough #Chain用的
from langchain_openai import AzureChatOpenAI,ChatOpenAI #創建大語言模型LLM用的
from flask import Flask, jsonify,request #後端API創建
from flask_cors import CORS #調用API相關
from dotenv import load_dotenv #使用.env管理相關變量

#創建應用
app = Flask(__name__)
CORS(app)

#加載根目錄的.env文件
load_dotenv()

#創建提示詞Prompt
template_prompt = """
你是一個髒話檢測工具，專門用於檢測一段內容裡是否含有髒話。
根據下面的"內容"，判定該內容是否有髒話，如果有髒話，請在"存在髒話"中輸出"是"，並在"髒話內容"中輸出該髒話句子。如果沒有髒話，則在"存在髒話"中輸出"否"，並在"髒話內容"中輸出"無"。

內容:{user_content}

你必須以以下格式返回輸出：
存在髒話：是/否
髒話內容：包含髒話的句子/無

例子：
內容：你他媽的幹哈呢？你是不是給臉不要臉？
存在髒話：是
髒話內容：你他媽的幹哈呢？
"""
my_prompt = ChatPromptTemplate.from_template(template_prompt)#此處將上面的String變成提示詞Prompt


#獲取.env變量
api_key = os.getenv("API_KEY")
model_name = os.getenv("MODEL_NAME")
database_url = os.getenv("DATABASE_URL")

#自填AI信息(推薦使用.env)
llm = ChatOpenAI(
    api_key= api_key,
    model = model_name,
    base_url = database_url
)

#將Prompt, LLM合為一體，並且將LLM的輸出格式化
chain = (
      RunnablePassthrough()
      | my_prompt
      | llm
      | StrOutputParser()
     )

#提取返回的內容，並且轉換為可讀的格式
def parse_ai_response(text):
    result = {}
    lines = text.split('\n')
    for line in lines:
        # 中文冒號分割鍵值，strip() 去除空白
        if '：' in line:
            key, value = line.split('：', 1)  # 只分割第一個冒號
            result[key.strip()] = value.strip()
    return result

#創建API路徑
@app.route('/')
def hello_world():
    return 'Hello World! 後端已成功運行'

@app.route('/api/get_ai_response', methods=['POST'])
def get_ai_response():
    data = request.json
    if not data or 'user_content' not in data:
        return jsonify({'error': 'Invalid Input'}), 400
    
    user_content = data['user_content']

    try:
        ai_response = chain.invoke({
            "user_content": user_content
        })
     
        parsed_data = parse_ai_response(ai_response)
        return jsonify(json.dumps(parsed_data, ensure_ascii=False, indent=2)), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

#啟動後端服務器，主機和端口可自定義。
if __name__ == '__main__':
    print("Server Running!")
    app.run(host="localhost",port= 23333,debug=True)

