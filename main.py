import io
import os
import pytesseract
import base64

from dotenv import load_dotenv
from flask import Flask, abort, request

from langchain_core.messages import SystemMessage
from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_openai import ChatOpenAI

from PIL import Image

app = Flask(__name__)


def prompt_llm(text):
    chat_messages = [
        SystemMessage(
            content=(
                "Du hilfst dabei Quizfragen aus verschiedenen Kategorien zu beantworten."
                "Dir wird ein String präsentiert, der die Frage sowie 4 mögliche Antworten enthält."
                "Der String enthält möglicherweise irrelevante Zeichen, da er aus einem Bild extrahiert wurde."
                "Gib als Anwort lediglich die korrekte Option zurück, es sind keine weiteren Erklärungen nötig."
                "Dies ist ein Beispiel: "
                "Wann wurde Elvis Presley geboren?"
                "8. Februar 1935"
                "13 August 1900"
                "21. März 1942"
                "8. Januar 1935"
                "Anwort:"
                "8. Januar 1935"
            )
        ),
    ]

    quiz_template = "Beantworte nun Folgende Frage: {question}"

    chat_messages.append(HumanMessagePromptTemplate.from_template(quiz_template))

    template = ChatPromptTemplate.from_messages(
        chat_messages,
    )

    model = ChatOpenAI(
        model="gpt-4-turbo",
    )
    chain = template | model

    api_response = chain.invoke({"question": text})

    return api_response.content


@app.route("/solve", methods=["POST"])
def solve():
    data = request.get_json()

    image_base_64 = data["question"]
    img_bytes = base64.b64decode(image_base_64)
    img = Image.open(io.BytesIO(img_bytes))

    text = pytesseract.image_to_string(img)

    load_dotenv()
    if "OPENAI_API_KEY" not in os.environ:
        abort(500)

    answer = prompt_llm(text)

    return answer


if __name__ == "__main__":
    app.run(debug=True, port=5000, host="0.0.0.0")
