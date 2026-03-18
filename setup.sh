#!/bin/bash

# ------------------------------------------------
# Verificar Python 3.11
# ------------------------------------------------
echo "Verificando Python 3.11..."

if command -v python3.11 >/dev/null 2>&1; then
    PYTHON_BIN=python3.11
elif command -v python3 >/dev/null 2>&1; then
    PY_VER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    if [ "$PY_VER" = "3.11" ]; then
        PYTHON_BIN=python3
    else
        echo "Python 3.11 nao encontrado. Instalando..."

        if command -v apt >/dev/null 2>&1; then
            sudo apt update
            sudo apt install -y python3.11 python3.11-venv
        elif command -v dnf >/dev/null 2>&1; then
            sudo dnf install -y python3.11
        elif command -v pacman >/dev/null 2>&1; then
            yay -S --noconfirm python311
        else
            echo "Gerenciador de pacotes nao suportado."
            echo "Instale manualmente o Python 3.11."
            exit 1
        fi

        PYTHON_BIN=python3.11
    fi
else
    echo "Python nao encontrado."
    exit 1
fi

echo "Usando $PYTHON_BIN"

echo "Iniciando setup do ambiente Label Studio"

VENV_DIR="./labelStudioVenv"
PYTHON_EXE="$VENV_DIR/bin/python"

# Se existir mas nao for executavel, recria a venv (ex: criada em outro SO)
if [ -e "$PYTHON_EXE" ] && [ ! -x "$PYTHON_EXE" ]; then
    echo "Ambiente virtual encontrado, mas o python nao e executavel. Recriando..."
    rm -rf "$VENV_DIR"
fi

# Criar venv se nao existir
if [ ! -x "$PYTHON_EXE" ]; then
    echo "Criando ambiente virtual..."
    "$PYTHON_BIN" -m venv "$VENV_DIR"

    if [ $? -ne 0 ]; then
        echo "ERRO ao criar venv."
        exit 1
    fi

    echo "Ambiente virtual criado com sucesso."
fi

# Garantir que estamos usando o Python da venv
if [ ! -x "$PYTHON_EXE" ]; then
    echo "ERRO: Python da venv nao encontrado."
    exit 1
fi

# Atualizar pip da venv
echo "Atualizando pip da venv..."
"$PYTHON_EXE" -m pip install --upgrade pip setuptools wheel

if [ $? -ne 0 ]; then
    echo "ERRO ao atualizar pip."
    exit 1
fi

# Instalar dependencias
echo "Instalando dependencias..."

LIBS=(
    tensorrt
    openvino
    label-studio
    label-studio-ml
    label-studio-sdk
    numpy
    opencv-python
    Pillow
    requests
    torch
    torchvision
    hydra-core
    omegaconf
    python-dotenv
    tqdm
    ultralytics
)

# Opcional: pular torch/torchvision (ex: TORCH_SKIP=1 ./setup.sh)
if [ "${TORCH_SKIP:-0}" = "1" ]; then
    for i in "${!LIBS[@]}"; do
        if [ "${LIBS[$i]}" = "torch" ] || [ "${LIBS[$i]}" = "torchvision" ]; then
            unset 'LIBS[i]'
        fi
    done
fi

echo -e "Será instalado as bibliotecas:\n \
    tensorrt openvino label-studio label-studio-ml label-studio-sdk\n \
    numpy opencv-python Pillow requests\n \
    torch torchvision hydra-core omegaconf\n \
    python-dotenv tqdm ultralytics\n"

printf "Se deseja NAO instalar alguma, digite as respectivas separando por ponto e virgula\n(ENTER para instalar todas): "
read -r DL_L

if [ -n "$DL_L" ]; then
    IFS=';' read -ra SKIP <<< "$DL_L"
    for LIB in "${SKIP[@]}"; do
        LIB=$(echo "$LIB" | xargs)  # remove espaços extras
        for i in "${!LIBS[@]}"; do
            if [ "${LIBS[$i]}" = "$LIB" ]; then
                unset 'LIBS[i]'
            fi
        done
    done
    echo ""
    echo "Bibliotecas a instalar: ${LIBS[*]}"
    echo ""
fi

PIP_CACHE_DIR="${PIP_CACHE_DIR:-$PWD/.pip_cache}"
mkdir -p "$PIP_CACHE_DIR"

# Se /tmp for tmpfs, use um diretorio no disco para temporarios do pip
if [ -z "${PIP_TMPDIR:-}" ]; then
    TMP_FSTYPE=$(findmnt -n -o FSTYPE /tmp 2>/dev/null)
    if [ "$TMP_FSTYPE" = "tmpfs" ]; then
        PIP_TMPDIR="$PWD/.pip_tmp"
    fi
fi

if [ -n "${PIP_TMPDIR:-}" ]; then
    mkdir -p "$PIP_TMPDIR"
fi

# Checagem simples de espaco em disco para evitar erro no meio do download
MIN_SPACE_MB="${MIN_SPACE_MB:-8000}"
check_space_mb() {
    local path="$1"
    df -Pm "$path" 2>/dev/null | awk 'NR==2 {print $4}'
}

TMP_CHECK_DIR="${PIP_TMPDIR:-${TMPDIR:-/tmp}}"
PIP_CACHE_FREE_MB=$(check_space_mb "$PIP_CACHE_DIR")
TMP_FREE_MB=$(check_space_mb "$TMP_CHECK_DIR")

if [ -n "$PIP_CACHE_FREE_MB" ] && [ "$PIP_CACHE_FREE_MB" -lt "$MIN_SPACE_MB" ]; then
    echo "ERRO: pouco espaco em $PIP_CACHE_DIR (${PIP_CACHE_FREE_MB} MB livres)."
    echo "Defina PIP_CACHE_DIR para um local com mais espaco ou use PIP_NO_CACHE=1."
    exit 1
fi

if [ -n "$TMP_FREE_MB" ] && [ "$TMP_FREE_MB" -lt "$MIN_SPACE_MB" ]; then
    echo "ERRO: pouco espaco em $TMP_CHECK_DIR (${TMP_FREE_MB} MB livres)."
    echo "Defina PIP_TMPDIR para um local com mais espaco."
    exit 1
fi

PIP_FLAGS=(--prefer-binary)
PIP_EXTRA_FLAGS=${PIP_EXTRA_FLAGS:-}
if [ "${PIP_NO_CACHE:-0}" != "1" ]; then
    PIP_FLAGS+=(--cache-dir "$PIP_CACHE_DIR")
fi

LIBS_LIST="${LIBS[*]}"
export PYTHON_EXE PIP_CACHE_DIR PIP_EXTRA_FLAGS LIBS_LIST
export TORCH_INDEX_URL="${TORCH_INDEX_URL:-}"
if [ -n "${PIP_TMPDIR:-}" ]; then
    export TMPDIR="$PIP_TMPDIR"
fi

if ! "$PYTHON_EXE" - <<'PY'
import os
import shlex
import subprocess
import sys

python_exe = os.environ["PYTHON_EXE"]
cache_dir = os.environ["PIP_CACHE_DIR"]
extra_flags = shlex.split(os.environ.get("PIP_EXTRA_FLAGS", ""))
torch_index_url = os.environ.get("TORCH_INDEX_URL", "").strip()
libs = os.environ.get("LIBS_LIST", "").split()

base_cmd = [python_exe, "-m", "pip", "install", "--cache-dir", cache_dir, "--prefer-binary"]

HEAVY_LIBS = {"torch", "torchvision", "tensorrt", "openvino"}

for lib in libs:
    cmd = base_cmd + extra_flags + [lib]
    if lib in HEAVY_LIBS:
        cmd = [python_exe, "-m", "pip", "install", "--no-cache-dir", "--prefer-binary"]
        cmd += extra_flags + [lib]
    if torch_index_url and lib in ("torch", "torchvision"):
        cmd += ["--index-url", torch_index_url]
    print(f"Instalando: {lib}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("")
        print("========================================")
        print(f"ERRO na instalacao das dependencias: {lib}")
        print("Verifique mensagens acima.")
        print("========================================")
        sys.exit(1)
PY
then
    exit 1
fi

# Limpa o cache após instalação (opcional)
if [ "${PIP_KEEP_CACHE:-0}" != "1" ]; then
    echo "Limpando cache do pip..."
    rm -rf "$PIP_CACHE_DIR"
    echo "Cache removido."
else
    echo "Cache mantido em $PIP_CACHE_DIR (PIP_KEEP_CACHE=1)."
fi

# Verificacao final real
echo
echo "Verificando instalacao do Label Studio..."

"$PYTHON_EXE" -c "import label_studio" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "ERRO: Label Studio nao foi instalado corretamente."
    exit 1
fi

echo
echo "========================================"
echo "Instalacao concluida com sucesso!"
echo "Utilize o run.sh para iniciar"
echo "========================================"
