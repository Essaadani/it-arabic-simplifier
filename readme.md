cd ~/projects
rm -rf it-arabic-simplifier
mkdir -p it-arabic-simplifier
cd it-arabic-simplifier

uname -m
python3 --version
xcode-select -p

python3 -m venv .venv
source .venv/bin/activate

which python
which pip

python -m pip install --upgrade pip setuptools wheel
python -m pip install "mlx-lm[train]"
python -m pip install pandas scikit-learn sentence-transformers sacrebleu rapidfuzz camel-tools streamlit

python -m pip show mlx-lm
python -m pip show mlx

python - <<'PY'
import mlx
import mlx_lm
print("MLX OK")
print("MLX-LM OK")
PY

python -m mlx_lm.generate --help
python -m mlx_lm.lora --help

python -m mlx_lm.generate \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --prompt "Explain in simple Arabic: A variable is a named location in memory used to store a value."

mkdir -p data/raw data/processed outputs/adapters outputs/eval scripts app
python -m pip freeze > requirements.txt
