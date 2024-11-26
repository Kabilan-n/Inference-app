FROM python:3.10
COPY . /webapp
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE $PORT
CMD gunicorn --workers=5 --bind 0.0.0.0:$PORT webapp:app