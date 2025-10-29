FROM mambaorg/micromamba:1.5.8

USER root
RUN apt-get update && apt-get install -y --no-install-recommends         r-base r-base-dev libcurl4-openssl-dev libssl-dev libxml2-dev         && rm -rf /var/lib/apt/lists/*

RUN R -e "install.packages('devtools', repos='https://cloud.r-project.org')" &&         R -e "install.packages(c('Rcpp','RcppArmadillo'), repos='https://cloud.r-project.org')" &&         R -e "if (!require('imbalance')) { devtools::install_github('unbalanced-parentheses/imbalance') }"

COPY environment.yml /tmp/environment.yml
RUN micromamba install -y -n base -f /tmp/environment.yml && micromamba clean --all --yes

WORKDIR /workspace
COPY . /workspace
RUN pip install -e ./coffee-imbalance

CMD ["bash", "-lc", "cd coffee-imbalance && make cv"]
