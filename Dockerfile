FROM python:3.11-slim

WORKDIR /app

# Install deps first — layer caching so code changes don't reinstall everything
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

#copy the rest of the project (respects .dockerignore)
COPY . .

EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]