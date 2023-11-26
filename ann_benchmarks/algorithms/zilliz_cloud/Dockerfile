FROM ann-benchmarks

WORKDIR /home/app
COPY run_algorithm.py ./

RUN pip3 install pymilvus
RUN python3 -c 'import pymilvus'

ENTRYPOINT ["python3", "-u", "run_algorithm.py"]