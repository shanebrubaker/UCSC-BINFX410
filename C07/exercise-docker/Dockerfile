FROM ubuntu:22.04

LABEL maintainer="BINFX410"
LABEL description="Common bioinformatics tools for sequence analysis"

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        samtools \
        bwa \
        fastqc \
        bedtools \
        ncbi-blast+ && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /data

CMD ["/bin/bash"]
