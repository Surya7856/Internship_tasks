FROM python:3.9-slim
WORKDIR /laptop_price_app
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
COPY . .
CMD ["streamlit","run","app.py"]

