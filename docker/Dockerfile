FROM vllm/vllm-openai:latest

WORKDIR /app

EXPOSE 8000 8001

ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server"]