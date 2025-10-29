# RoastMasterAI

## Quickstart

### Conda
```bash
cd coffee-imbalance
conda env create -f environment.yml
conda activate coffee-imbalance
pip install -e .
make setup
```

### Docker (RACOG 포함)
```bash
docker build -t roastai:latest -f coffee-imbalance/Dockerfile .
docker run --rm -it -v $PWD:/workspace roastai:latest
```

## One-liners
```bash
cd coffee-imbalance
make features
make validate
make cv
make grid
make final
make external
make report
```
