FROM tensorflow/tensorflow:2.16.1-gpu

WORKDIR /workspace

COPY docker/requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r /workspace/requirements.txt

COPY train /workspace/train

CMD ["bash", "/workspace/train/run_all.sh"]