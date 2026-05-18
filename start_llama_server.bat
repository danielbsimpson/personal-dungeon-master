@echo off
:: ─────────────────────────────────────────────────────────────────────────────
:: start_llama_server.bat — Launch llama-server for use with Personal DM
::
:: Prerequisites:
::   1. Download llama-*-win-cuda-cu12.x.x-x64.zip from
::      https://github.com/ggml-org/llama.cpp/releases
::   2. Extract and add llama-server.exe to PATH, or set LLAMA_EXE below.
::   3. Place your GGUF model at the path set in LLAMA_MODEL_PATH.
:: ─────────────────────────────────────────────────────────────────────────────

:: ── Configuration — edit these three lines ───────────────────────────────────
set LLAMA_EXE=C:\llama.cpp\llama-server.exe
set LLAMA_MODEL_PATH=%USERPROFILE%\.ollama\models\blobs\sha256-dde5aa3fc5ffc17176b5e8bdc82f587b24b2678c6c66101bf7da77af9f7ccdff
set LLAMA_ALIAS=llama3.2-3b

:: ── Server options ────────────────────────────────────────────────────────────
set LLAMA_PORT=8080
set LLAMA_HOST=127.0.0.1
set LLAMA_GPU_LAYERS=999
:: Context size — increase for longer conversations (uses more VRAM)
set LLAMA_CTX_SIZE=8192

:: ─────────────────────────────────────────────────────────────────────────────
echo Starting llama-server...
echo   Model : %LLAMA_MODEL_PATH%
echo   Alias : %LLAMA_ALIAS%
echo   Port  : %LLAMA_HOST%:%LLAMA_PORT%
echo   GPU   : --n-gpu-layers %LLAMA_GPU_LAYERS%
echo.

%LLAMA_EXE% ^
  --model "%LLAMA_MODEL_PATH%" ^
  --alias "%LLAMA_ALIAS%" ^
  --n-gpu-layers %LLAMA_GPU_LAYERS% ^
  --ctx-size %LLAMA_CTX_SIZE% ^
  --port %LLAMA_PORT% ^
  --host %LLAMA_HOST%

:: Non-zero exit means the server crashed or the exe was not found.
if %ERRORLEVEL% neq 0 (
  echo.
  echo ERROR: llama-server exited with code %ERRORLEVEL%.
  echo Make sure llama-server.exe is in PATH or update LLAMA_EXE above.
  pause
)
