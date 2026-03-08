FROM python:3.10-slim

# Install Java (required for Spark)
RUN apt-get update && \
    apt-get install -y --no-install-recommends default-jre-headless && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set Java and Spark environment variables
ENV JAVA_HOME=/usr
ENV SPARK_HOME=/usr/local/lib/python3.10/site-packages/pyspark
ENV PATH="${SPARK_HOME}/bin:${PATH}"

# Install Python packages
RUN pip install --no-cache-dir \
    pyspark==3.5.0 \
    jupyter==1.0.0 \
    jupyterlab==4.0.0 \
    pandas==2.1.0 \
    matplotlib==3.8.0 \
    seaborn==0.13.0 \
    numpy==1.26.0 \
    pyarrow==14.0.0 \
    tqdm

# Create non-root user
RUN useradd -m -u 1000 student

# Set working directory
WORKDIR /workspace

# Scripts, notebooks, and data are mounted via docker-compose volumes.
# Create directories so permissions are correct even before volumes mount.
RUN mkdir -p /workspace/data /workspace/notebooks /workspace/scripts && \
    chown -R student:student /workspace && \
    chmod -R 755 /workspace

# Expose Jupyter and Spark UI ports
EXPOSE 8888
EXPOSE 4040

# Switch to non-root user
USER student

# Launch Jupyter Lab
ENTRYPOINT ["jupyter", "lab", \
    "--ip=0.0.0.0", \
    "--port=8888", \
    "--no-browser", \
    "--NotebookApp.token=''"]
