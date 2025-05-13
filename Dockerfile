# ===== Stage 1: Builder =====
FROM python:3.10-slim as builder

WORKDIR /app

COPY requirement.txt .

# Install dependencies and freeze them
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirement.txt \
 && pip freeze > frozen-requirements.txt

# ===== Stage 2: Runtime =====
FROM python:3.10-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy the application files
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "sales_forecasting_app.py", "--server.port=8501"]