import json
import logging
import os
from typing import List

import colorama
import requests
from colorama import Fore, Style
from tqdm import tqdm

# Import centralized settings (paths, constants)
from settings import (
    DOWNLOADS_SUBDIR,
    OUTPUT_SUBDIR,
    DEFAULT_START_VERSION,
    HELM_VERSIONS,
    HELM_URL_WITH_BENCHMARK_TEMPLATE,
    HELM_URL_WITHOUT_BENCHMARK_TEMPLATE,
    HELM_FILE_TYPES,
)

# Initialize colorama
colorama.init()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("helm_processor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("HELM_Processor")


# Custom logger function with emojis and colors
def log_info(message, emoji="ℹ️"):
    logger.info(f"{emoji} {message}")
    print(f"{Fore.CYAN}{emoji} {message}{Style.RESET_ALL}")


def log_success(message, emoji="✅"):
    logger.info(f"{emoji} {message}")
    print(f"{Fore.GREEN}{emoji} {message}{Style.RESET_ALL}")


def log_error(message, emoji="❌"):
    logger.error(f"{emoji} {message}")
    print(f"{Fore.RED}{emoji} {message}{Style.RESET_ALL}")


def log_warning(message, emoji="⚠️"):
    logger.warning(f"{emoji} {message}")
    print(f"{Fore.YELLOW}{emoji} {message}{Style.RESET_ALL}")


def log_step(step_name, emoji="🔄"):
    logger.info(f"{emoji} {step_name}")
    print(f"{Fore.MAGENTA}{emoji} {step_name}{Style.RESET_ALL}")


# Configure base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOWNLOADS_DIR = os.path.join(BASE_DIR, DOWNLOADS_SUBDIR)
OUTPUT_DIR = os.path.join(BASE_DIR, OUTPUT_SUBDIR)

# Create necessary directories
os.makedirs(DOWNLOADS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_json_from_url(url):
    """Fetch JSON data from given URL."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        # log_error(f"An error occurred with URL {url}: {e}")
        return None


def download_task(task: str, output_dir: str, benchmark: str, overwrite: bool = False,
                  start_version: str = DEFAULT_START_VERSION) -> str or None:
    """
    Download and extract files for a given task.
    This function tries different versions of HELM data.
    """
    log_step(f"Downloading task: {task}", "🔽")

    # List of versions to check, in order
    versions = list(HELM_VERSIONS)  # v1.0.0 to v1.13.0
    # Start from the specified version
    start_idx = versions.index(start_version)
    versions = versions[start_idx:]

    # Base URL template
    base_url_template = HELM_URL_WITH_BENCHMARK_TEMPLATE

    # Different file types to download for each task
    file_types = list(HELM_FILE_TYPES)

    # Create directory for this task with proper permissions for later deletion
    save_dir = os.path.join(output_dir, task)

    # Remove directory if it exists to ensure clean state
    if os.path.exists(save_dir) and overwrite:
        try:
            import shutil
            shutil.rmtree(save_dir)
            log_info(f"Removed existing directory: {save_dir}", "🗑️")
        except Exception as e:
            log_warning(f"Could not remove existing directory: {e}", "⚠️")

    # Create the directory with full permissions
    os.makedirs(save_dir, exist_ok=True)

    # Set directory permissions to 0o777 (rwxrwxrwx) to ensure it can be deleted later
    try:
        os.chmod(save_dir, 0o777)
        log_info(f"Set directory permissions to allow deletion", "🔑")
    except Exception as e:
        log_warning(f"Could not set directory permissions: {e}", "⚠️")

    # Initialize statistics for this task
    task_stats = {
        "task": task,
        "total_files": len(file_types),
        "found_files": 0,
        "missing_files": 0,
        "version_usage": {v: 0 for v in versions}
    }

    for file_type in file_types:
        save_path = os.path.join(save_dir, f"{file_type}.json")
        found = False

        # Skip if file exists and not overwriting
        if os.path.exists(save_path) and not overwrite:
            log_info(f"File {file_type}.json already exists for task {task} (skipping)", "📄")
            task_stats["found_files"] += 1
            continue

        # Try each version in order until we find the file
        for version in versions:
            cur_url = f"{base_url_template.format(benchmark=benchmark, version=version)}/{task}/{file_type}.json"
            json_data = get_json_from_url(cur_url)
            if json_data is None:
                cur_url = f"{HELM_URL_WITHOUT_BENCHMARK_TEMPLATE.format(version=version)}/{task}/{file_type}.json"
                json_data = get_json_from_url(cur_url)

            if json_data:
                with open(save_path, "w") as f:
                    json.dump(json_data, f, indent=2)
                task_stats["version_usage"][version] += 1
                found = True
                task_stats["found_files"] += 1
                log_success(f"Found {task}/{file_type}.json in version {version}", "📥")
                break

        if not found:
            task_stats["missing_files"] += 1
            log_warning(f"Could not find {task}/{file_type}.json in any version", "🔍")

    # Log success/failure info
    if task_stats["found_files"] == task_stats["total_files"]:
        log_success(f"Downloaded all {task_stats['total_files']} files for task {task}", "🎉")
    else:
        log_warning(
            f"Downloaded {task_stats['found_files']}/{task_stats['total_files']} files for task {task}",
            "⚠️"
        )

    return save_dir


def download_tasks(tasks: List[str], output_dir: str, benchmark: str, overwrite: bool = False,
                   start_version: str = DEFAULT_START_VERSION) -> List[str]:
    """
    Download a list of tasks and return the paths to the saved files.
    """
    log_step(f"Downloading {len(tasks)} tasks", "🔽")

    saved_files = []
    for task in tqdm(tasks, desc="Processing tasks"):
        log_step(f"Starting download for task: {task}")
        saved_file = download_task(task, output_dir, benchmark, overwrite, start_version)
        if saved_file:
            saved_files.append(saved_file)
            log_success(f"Successfully downloaded task '{task}' to '{saved_file}'")
        else:
            log_error(f"Failed to download task: {task}")

    return saved_files
