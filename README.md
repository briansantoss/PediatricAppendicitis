# Instruções de uso

## 1. Ativando o ambiente virtual

No terminal, navegue até a pasta do projeto e execute o comando apropriado:

### 🔷 Windows

Se estiver usando **PowerShell**:
```powershell
.venv\Scripts\Activate.ps1
```

Caso esteja usando o **Cmd**
```
.venv\Scripts\activate.bat
```

### 🐧 Linux / macOS (sistemas POSIX)

```bash
source .venv/bin/activate
```

> Após isso, o nome do ambiente virtual deve aparecer no início da linha de comando, indicando que ele está ativo.

----

### 💡 Dica extra

Se o ambiente virtual não for ativado no PowerShell, pode ser por restrições de segurança. Execute este comando **uma vez** com permissões de administrador:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

## 2. Instale as dependências

Com o ambiente ativado execute

```bash
pip install -r requirements.txt
```

### 3. Execute o projeto

Execute

```bash
python main.py
```
