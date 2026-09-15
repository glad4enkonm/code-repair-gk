#!/usr/bin/env python3
"""
BM25 baseline — inference pipeline.

Загружает датасет, для каждого инстанса:
  1. Клонирует репозиторий (зеркало swe-bench)
  2. Строит BM25-индекс по исходникам
  3. Ищет top-k релевантных файлов к тексту issue
  4. Формирует промпт (style-2) и вызывает Qwen через llama.cpp API
  5. Извлекает diff из ответа, сохраняет в predictions.jsonl

Запуск:
    OPENAI_API_KEY=any OPENAI_BASE_URL=http://localhost:8080/v1 \
        python3 src/inference/run_qwen.py
"""

import json
import logging
import os
import sys
import tempfile
import traceback
from pathlib import Path

from datasets import load_from_disk
from git import Repo
from openai import OpenAI
from tqdm.auto import tqdm

# готовые блоки из swebench
from swebench.inference.make_datasets.bm25_retrieval import (
    DOCUMENT_ENCODING_FUNCTIONS,
    make_index,
    search,
)
from swebench.inference.make_datasets.create_instance import prompt_style_2
from swebench.inference.make_datasets.utils import (
    ContextManager,
    extract_diff,
)

logger = logging.getLogger(__name__)

MIRROR_URL = "https://github.com/swe-bench-repos/{repo}.git"


def clone_repo(repo: str, root_dir: str) -> str:
    """Клонирует репозиторий из официального зеркала SWE-bench."""
    repo_name = repo.replace("/", "__")
    repo_dir = Path(root_dir) / repo_name

    if not repo_dir.exists():
        url = MIRROR_URL.format(repo=repo_name)
        print(f"  Cloning {repo} ...", end=" ", flush=True)
        Repo.clone_from(url, repo_dir, depth=1)
        print("OK")
    else:
        print(f"  Using cached {repo}")
    return str(repo_dir)


def build_prompt(problem_statement: str, file_contents: dict, readmes: dict) -> str:
    """Формирует промпт в стиле style-2 (с тегами <issue>, <code>)."""
    return prompt_style_2(
        {
            "problem_statement": problem_statement,
            "readmes": readmes,
            "file_contents": file_contents,
        }
    )


def call_model(client, config: dict, prompt: str) -> str:
    """Отправляет промпт в Qwen через OpenAI-совместимый API."""
    # Разделяем на system/user сообщения (как в run_api.py)
    parts = prompt.split("\n", 1)
    system_msg = parts[0]
    user_msg = parts[1] if len(parts) > 1 else ""

    response = client.chat.completions.create(
        model=config["model"],
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=config["temperature"],
        max_tokens=config["max_tokens"],
    )
    return response.choices[0].message.content


def process_instance(client, config: dict, instance: dict) -> dict | None:
    """Обрабатывает один инстанс: клонирование → BM25 → промпт → Qwen → patch."""
    instance_id = instance["instance_id"]
    repo = instance["repo"]
    base_commit = instance["base_commit"]
    problem_statement = instance["problem_statement"]

    encoding_func = DOCUMENT_ENCODING_FUNCTIONS[config["encoding"]]
    k = config["top_k"]

    with tempfile.TemporaryDirectory() as root_dir:
        # 1. Клонируем репозиторий
        repo_dir = clone_repo(repo, root_dir)

        # 2. Строим BM25-индекс
        index_path = make_index(
            repo_dir=repo_dir,
            root_dir=root_dir,
            query=problem_statement,
            commit=base_commit,
            document_encoding_func=encoding_func,
            python=sys.executable,
            instance_id=instance_id,
        )

        # 3. BM25-поиск релевантных файлов
        result = search(instance, index_path)
        if result is None or not result.get("hits"):
            print(f"  No hits for {instance_id}")
            return None

        hits = result["hits"][:k]
        file_paths = [h["docid"] for h in hits]

        # 4. Читаем содержимое найденных файлов и README
        with ContextManager(repo_dir, base_commit):
            file_contents = {}
            for fp in file_paths:
                full_path = os.path.join(repo_dir, fp)
                if os.path.exists(full_path):
                    with open(full_path, encoding="utf-8", errors="replace") as f:
                        file_contents[fp] = f.read()

            readmes = {}
            readme_names = ContextManager(repo_dir, base_commit)
            with readme_names as cm:
                for rp in cm.get_readme_files():
                    full_path = os.path.join(repo_dir, rp)
                    with open(full_path, encoding="utf-8", errors="replace") as f:
                        readmes[rp] = f.read()

        if not file_contents:
            print(f"  No file contents for {instance_id}")
            return None

        # 5. Промпт и вызов модели
        prompt = build_prompt(problem_statement, file_contents, readmes)
        response_text = call_model(client, config, prompt)
        model_patch = extract_diff(response_text)

        return {
            "instance_id": instance_id,
            "model_name_or_path": config["model"],
            "model_patch": model_patch,
        }


def load_existing(output_file: Path) -> set:
    """Загружает ID уже обработанных инстансов (для возобновления)."""
    if not output_file.exists():
        return set()
    ids = set()
    with open(output_file) as f:
        for line_no, line in enumerate(f, 1):
            try:
                ids.add(json.loads(line)["instance_id"])
            except (json.JSONDecodeError, KeyError) as exc:
                logger.warning(
                    "skipping malformed line %d in %s: %s",
                    line_no,
                    output_file,
                    exc,
                )
    return ids


def main():
    # Конфиг: можно задать путь через аргумент или переменную окружения
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/qwen_config.json"
    with open(config_path) as f:
        config = json.load(f)

    # OpenAI-совместимый клиент → llama.cpp
    client = OpenAI(
        base_url=os.environ.get("OPENAI_BASE_URL", config["api_base"]),
        api_key=os.environ.get("OPENAI_API_KEY", "not-needed"),
    )

    # Загружаем датасет
    dataset = load_from_disk(config["dataset_path"])
    split = config["split"]
    instances = dataset[split]
    print(f"Dataset: {config['dataset_path']}/{split} — {len(instances)} instances")

    # Выходной файл
    output_file = Path(config["output_dir"]) / f"predictions_{split}.jsonl"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    existing_ids = load_existing(output_file)
    if existing_ids:
        print(f"Resuming: {len(existing_ids)} already processed, will skip")

    # Основной цикл
    for instance in tqdm(instances, desc=f"Running BM25 baseline on {split}"):
        instance_id = instance["instance_id"]
        if instance_id in existing_ids:
            continue

        print(f"\n[{instance_id}] {instance['repo']}")
        try:
            prediction = process_instance(client, config, instance)
            if prediction is None:
                continue

            with open(output_file, "a") as f:
                print(json.dumps(prediction), file=f, flush=True)
            existing_ids.add(instance_id)

        except Exception:
            print(f"  FAILED: {instance_id}")
            traceback.print_exc()

    # Итоги
    with open(output_file) as f:
        total = sum(1 for _ in f)
    print(f"\nDone. {total} predictions → {output_file}")


if __name__ == "__main__":
    main()
