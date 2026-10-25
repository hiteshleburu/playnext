FROM python:3.13-slim

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    WEB_SEARCH_CACHE=/tmp/web_search
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user src ./src
COPY --chown=user web ./web
COPY --chown=user data/games_slim.pkl ./data/games_slim.pkl

EXPOSE 7860
CMD ["uvicorn", "api:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "7860"]