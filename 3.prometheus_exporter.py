from prometheus_client import Counter, Histogram, Gauge, Summary
import psutil
import time

REQUEST_COUNT = Counter(
    'prediction_requests_total',
    'Total Prediction Requests',
    ['method', 'endpoint', 'status']
)

PREDICTION_LATENCY = Histogram(
    'prediction_latency_seconds',
    'Prediction Latency in seconds',
    ['endpoint']
)

MODEL_ACCURACY = Gauge(
    'model_accuracy',
    'Current Model Accuracy'
)

CPU_USAGE = Gauge(
    'cpu_usage_percent',
    'Current CPU Usage Percentage'
)

MEMORY_USAGE = Gauge(
    'memory_usage_percent',
    'Current Memory Usage Percentage'
)

FRAUD_PREDICTIONS = Counter(
    'fraud_predictions_total',
    'Total Fraud Predictions'
)

NORMAL_PREDICTIONS = Counter(
    'normal_predictions_total',
    'Total Normal Predictions'
)

PREDICTION_CONFIDENCE = Histogram(
    'prediction_confidence',
    'Prediction Confidence Score',
    ['prediction_class']
)

API_RESPONSE_TIME = Summary(
    'api_response_time_seconds',
    'API Response Time in seconds'
)

REQUEST_ERRORS = Counter(
    'request_errors_total',
    'Total Request Errors',
    ['error_type']
)

MODEL_PREDICTION_DRIFT = Gauge(
    'model_prediction_drift_score',
    'Model Prediction Drift Score'
)

THROUGHPUT = Gauge(
    'requests_per_second',
    'Current Requests Per Second'
)

DISK_USAGE = Gauge(
    'disk_usage_percent',
    'Current Disk Usage Percentage'
)

NETWORK_IO_BYTES = Counter(
    'network_io_bytes_total',
    'Total Network IO in bytes',
    ['direction']
)


def update_system_metrics():
    CPU_USAGE.set(psutil.cpu_percent(interval=0.1))
    MEMORY_USAGE.set(psutil.virtual_memory().percent)
    DISK_USAGE.set(psutil.disk_usage('/').percent)
    net_io = psutil.net_io_counters()
    NETWORK_IO_BYTES.labels(direction='sent').inc(net_io.bytes_sent)
    NETWORK_IO_BYTES.labels(direction='recv').inc(net_io.bytes_recv)