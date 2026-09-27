from prometheus_client import start_http_server, Gauge
import pandas as pd
import time


# Read benchmark results
df = pd.read_csv("../../evaluation/results.csv")


# Metrics
accuracy = Gauge(
    "rfp_model_accuracy",
    "Model accuracy percentage",
    ["model"]
)

latency = Gauge(
    "rfp_model_latency",
    "Model latency in seconds",
    ["model"]
)

tokens = Gauge(
    "rfp_model_tokens",
    "Total tokens used",
    ["model"]
)


# Export results
for _, row in df.iterrows():

    model = row["model"]

    accuracy.labels(
        model=model
    ).set(row["field_accuracy_pct"])


    latency.labels(
        model=model
    ).set(row["latency_seconds"])


    tokens.labels(
        model=model
    ).set(row["total_tokens"])


# Start Prometheus server
start_http_server(9101)

print("RFP Benchmark Metrics Exporter is running on port 8000")


while True:
    time.sleep(10)
