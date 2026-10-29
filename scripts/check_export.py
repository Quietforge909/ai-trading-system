"""Verify selected-source identity and bounded export privacy checks."""
import ast
import hashlib
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]
manifest=json.loads((ROOT/'docs/source-manifest.json').read_text())
problems=[]
for row in manifest['source_files']:
    path=ROOT/row['path']
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:
        problems.append(row['path']+': original source identity differs')

for path in ROOT.rglob('*'):
    if not path.is_file() or any(part in {'.git','__pycache__','.venv','venv'} for part in path.relative_to(ROOT).parts):continue
    rel=path.relative_to(ROOT).as_posix()
    if path.suffix in {'.pt','.pth','.cbm','.parquet','.pem','.key','.log','.pkl','.safetensors'}:
        problems.append(rel+': private artifact type')
    if path.name.startswith('.env') and path.name!='.env.example':problems.append(rel+': environment file')
    text=path.read_text()
    if re.search(r'^-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----',text,re.M):problems.append(rel+': private key pattern')
    if re.search(r'https://(?:discord(?:app)?\.com)/api/webhooks/\d+/[^\s\"\']+',text):problems.append(rel+': webhook pattern')
    if re.search(r'/(?:home|Users)/[A-Za-z0-9_.-]+/',text):problems.append(rel+': personal absolute path')
    if path.suffix=='.py':
        tree=ast.parse(text,filename=rel)
        for node in ast.walk(tree):
            if not isinstance(node,(ast.Assign,ast.AnnAssign)):continue
            value=node.value
            if not isinstance(value,ast.Constant) or not isinstance(value.value,str) or not value.value:continue
            targets=node.targets if isinstance(node,ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target,ast.Name) and re.fullmatch(r'(?:api_?key|api_?secret|password|access_?token|discord_?webhook_?url)',target.id,re.I):
                    problems.append(f'{rel}:{node.lineno}: nonempty credential literal')

print(json.dumps({'original_source_files':len(manifest['source_files']),'problems':problems,'remote_publication_performed':False},indent=2))
raise SystemExit(bool(problems))
