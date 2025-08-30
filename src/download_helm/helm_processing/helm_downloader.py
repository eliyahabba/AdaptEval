import json
import logging
import os
from typing import List, Dict

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
    HELM_LITE_BASE_URL_TEMPLATE,
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


def download_single_task(task: str, start_version: str = DEFAULT_START_VERSION, output_dir: str = DOWNLOADS_DIR,
                         overwrite: bool = False) -> Dict:
    """
    Download a single HELM task from multiple possible versions
    Returns statistics about what was downloaded
    """
    log_step(f"Downloading task: {task}", "🔽")

    # List of versions to check, in order
    versions = list(HELM_VERSIONS)  # v1.0.0 to v1.13.0
    # Start from the specified version
    start_idx = versions.index(start_version)
    versions = versions[start_idx:]

    # Base URL template
    base_url_template = HELM_LITE_BASE_URL_TEMPLATE

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
            cur_url = f"{base_url_template.format(version=version)}/{task}/{file_type}.json"
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

    return task_stats


def download_tasks(tasks: List[str], start_version: str = DEFAULT_START_VERSION,
                   output_dir: str = DOWNLOADS_DIR, overwrite: bool = False) -> Dict:
    """
    Download multiple HELM tasks
    Returns statistics about what was downloaded
    """
    log_step(f"Downloading {len(tasks)} tasks", "🔽")

    # Initialize statistics
    download_stats = {
        "total_tasks": len(tasks),
        "total_files_checked": len(tasks) * len(HELM_FILE_TYPES),
        "files_found": 0,
        "files_missing": 0,
        "version_usage": {version: 0 for version in HELM_VERSIONS}
    }

    # Process each task with a progress bar
    for task in tqdm(tasks, desc="Processing tasks"):
        task_stats = download_single_task(task, start_version, output_dir, overwrite)

        # Update global statistics
        download_stats["files_found"] += task_stats["found_files"]
        download_stats["files_missing"] += task_stats["missing_files"]

        # Update version usage
        for version, count in task_stats["version_usage"].items():
            if version in download_stats["version_usage"]:
                download_stats["version_usage"][version] += count

    # Print final statistics
    log_step("Download Statistics", "📊")
    log_info(f"Total tasks processed: {download_stats['total_tasks']}")
    log_info(f"Total files checked: {download_stats['total_files_checked']}")
    log_info(f"Files found: {download_stats['files_found']}")
    log_info(f"Files missing: {download_stats['files_missing']}")

    log_step("Files found per version:", "📈")
    for version, count in download_stats["version_usage"].items():
        if count > 0:
            log_info(f"{version}: {count} files")

    return download_stats
