import json
import logging
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List

import colorama
import pandas as pd
# Create a colorful progress bar
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

# Import centralized settings for paths and constants
from settings import (
    DOWNLOADS_SUBDIR,
    OUTPUT_SUBDIR,
    TQDM_BAR_FORMAT,
    PROCESS_POOL_MAX_WORKERS,
    DEFAULT_CSV_FILE_PROCESSOR,
)
# Import functions from helm_downloader.py
from helm_downloader import (
    download_tasks, log_info, log_success, log_error, log_warning, log_step
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

# Configure base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOWNLOADS_DIR = os.path.join(BASE_DIR, DOWNLOADS_SUBDIR)
OUTPUT_DIR = os.path.join(BASE_DIR, OUTPUT_SUBDIR)

# Create necessary directories
os.makedirs(DOWNLOADS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def read_tasks_from_csv(csv_file: str, adapter_method: str = None) -> List[str]:
    """
    Read tasks from CSV file in the format:
    ,Run,Model,Groups,Adapter method,Subject / Task

    Parameters:
        csv_file: Path to the CSV file
        adapter_method: Optional filter value for the 'Adapter method' column
                       (e.g., 'multiple_choice_joint')

    Returns a list of Run values (HELM catalog lines)
    """
    try:
        df = pd.read_csv(csv_file)
        total_rows = len(df)
        log_info(f"CSV file contains {total_rows} total rows", "📊")

        if 'Run' not in df.columns:
            raise ValueError(f"CSV file {csv_file} doesn't have a 'Run' column")

        # Apply filter if adapter_method is provided
        if adapter_method and 'Adapter method' in df.columns:
            log_info(f"Filtering tasks by Adapter method: {adapter_method}", "🔍")
            filtered_df = df[df['Adapter method'] == adapter_method]
            filtered_rows = len(filtered_df)

            # Log the filtering results
            log_info(
                f"Found {filtered_rows} tasks with Adapter method '{adapter_method}' out of {total_rows} total tasks",
                "📋")
            log_info(f"Using {filtered_rows}/{total_rows} rows ({filtered_rows / total_rows * 100:.1f}%)", "📊")

            if filtered_rows == 0:
                log_warning(f"No tasks found with Adapter method '{adapter_method}'", "⚠️")
                # Show available adapter methods for debugging
                available_methods = df['Adapter method'].unique()
                log_info(f"Available Adapter methods: {', '.join(str(m) for m in available_methods)}", "ℹ️")

            df = filtered_df

        tasks_list = list(df['Run'])
        valid_tasks = [task for task in tasks_list if task and isinstance(task, str)]

        log_info(f"Read {len(valid_tasks)} valid tasks from {csv_file}", "📄")
        if len(valid_tasks) < len(tasks_list):
            log_warning(f"Filtered out {len(tasks_list) - len(valid_tasks)} invalid or empty tasks", "⚠️")

        return valid_tasks
    except Exception as e:
        log_error(f"Error reading CSV file {csv_file}: {str(e)}", "❌")
        return []


def convert_data(data_dir: str) -> str:
    """
    Convert the extracted data using convert_cluade.py
    Returns the path to the converted CSV file
    """
    # Create a unique output filename based on the data directory
    dir_basename = os.path.basename(data_dir)
    output_file = os.path.join(OUTPUT_DIR, f"{dir_basename}_converted.csv")

    log_step(f"Converting data from {data_dir} to {output_file}", "🔄")

    try:
        # Import the helm_converter module directly
        from src.download_helm.helm_processing.helm_converter import main as convert_main

        # Call the main function directly
        log_info(f"Running conversion", "⚙️")
        convert_main(data_dir, output_file)

        log_success(f"Conversion completed successfully")
        return output_file

    except ImportError as e:
        log_error(f"Error importing helm_converter: {e}")
        return None


def cleanup(zip_path: str) -> None:
    """
    Clean up temporary files and directories
    """
    log_step(f"Cleaning up temporary files", "🧹")

    # Remove zip file
    if os.path.exists(zip_path):
        shutil.rmtree(zip_path) if os.path.isdir(zip_path) else os.remove(zip_path)
        log_info(f"Removed ZIP file: {zip_path}", "🗑️")


def process_line(line: str, keep_temp_files: bool = False, overwrite: bool = False) -> dict:
    """
    Process a single line from start to finish
    Returns a dictionary with the results
    """
    log_step(f"Processing line: {line}", "🔄")

    result = {
        "input_line": line,
        "status": "skipped",  # Default to skipped
        "downloaded_file": None,
        "converted_file": None,
        "entry_count": 0,
        "error": None
    }

    # Create a predictable output filename based on the line
    expected_output_file = os.path.join(OUTPUT_DIR, f"{line}_converted.csv")

    # Check if the output file already exists
    if os.path.exists(expected_output_file) and not overwrite:
        log_info(f"Output file already exists: {expected_output_file}", "📋")

        # Count rows in the existing CSV file
        try:
            df_existing = pd.read_csv(expected_output_file)
            entry_count = len(df_existing)
            result["entry_count"] = entry_count
            log_info(f"File contains {entry_count} rows", "🔢")
        except Exception as e:
            log_warning(f"Could not count rows in existing CSV: {e}", "⚠️")

        log_success(f"Skipping processing for line: {line} (output file already exists)", "⏭️")
        result["converted_file"] = expected_output_file
        return result

    # If we get here, we need to process the line
    result["status"] = "failed"  # Reset status to failed for processing

    try:
        # Download the data
        log_step(f"Starting download phase", "🔽")
        downloaded_files = download_tasks([line], output_dir=DOWNLOADS_DIR, overwrite=overwrite)
        if not downloaded_files:
            log_error(f"No files were downloaded")
            raise ValueError("No files were downloaded")

        saved_dir = os.path.join(DOWNLOADS_DIR, line)

        # Convert the data
        log_step(f"Starting conversion phase", "🔄")
        converted_file = convert_data(saved_dir)
        if not converted_file:
            log_error(f"Conversion failed")
            raise ValueError("Conversion failed")

        result["converted_file"] = converted_file

        # Count rows in the converted CSV file
        try:
            df_converted = pd.read_csv(converted_file)
            entry_count = len(df_converted)
            result["entry_count"] = entry_count
            log_info(f"Converted file contains {entry_count} rows", "🔢")
        except Exception as e:
            log_warning(f"Could not count rows in converted CSV: {e}", "⚠️")

        log_success(f"Conversion phase completed: {converted_file}", "📄")

        # Clean up if not keeping temp files
        if not keep_temp_files:
            log_step(f"Starting cleanup phase", "🧹")
            cleanup(saved_dir)
            log_success(f"Cleanup phase completed", "🧹")
        else:
            log_info(f"Skipping cleanup (keep_temp_files=True)", "🚫")

        result["status"] = "success"
        log_success(f"Processing completed successfully for line: {line}", "🎉")

    except Exception as e:
        log_error(f"Error processing line '{line}': {str(e)}")
        result["error"] = str(e)

    return result


def main(input_file: str = None, input_lines: List[str] = None, csv_file: str = None,
         adapter_method: str = None, keep_temp_files: bool = False, overwrite: bool = False):
    """
    Main function to process all lines
    """
    # Process input from CSV, file, or list
    lines = read_tasks_from_csv(csv_file, adapter_method)

    if not lines:
        log_warning("No tasks to process after filtering", "⚠️")
        return []

    results = []
    total_entries = 0

    # Configure colors for the progress bar
    bar_format = TQDM_BAR_FORMAT

    # Use tqdm with logging redirect to prevent progress bar disruption by log messages
    with logging_redirect_tqdm():
        # Create the progress bar
        pbar = tqdm(
            total=len(lines),
            desc=f"Processing {len(lines)} tasks",
            bar_format=bar_format,
            colour='green',
            ncols=100
        )

        # Parallel execution with a fixed-size process pool
        with ProcessPoolExecutor(max_workers=PROCESS_POOL_MAX_WORKERS) as executor:
            future_to_line = {executor.submit(process_line, line, keep_temp_files, overwrite): line for line in lines}

            for future in as_completed(future_to_line):
                line = future_to_line[future]
                try:
                    result = future.result()
                except Exception as e:
                    log_error(f"Error processing line '{line}': {str(e)}")
                    result = {
                        "input_line": line,
                        "status": "failed",
                        "downloaded_file": None,
                        "converted_file": None,
                        "entry_count": 0,
                        "error": str(e)
                    }

                results.append(result)
                total_entries += result.get("entry_count", 0)

                # Log the result
                if result["status"] == "success":
                    status_emoji = "✅"
                    pbar.colour = 'green'
                elif result["status"] == "skipped":
                    status_emoji = "⏭️"
                    pbar.colour = 'blue'
                else:
                    status_emoji = "❌"
                    pbar.colour = 'red'

                # Update progress bar with status and entry count
                pbar.set_postfix(
                    status=result["status"],
                    entries=result.get("entry_count", 0),
                    total_entries=total_entries
                )

                log_info(f"{status_emoji} Completed processing: {line} ({result.get('entry_count', 0)} entries)")

                # Update the progress bar
                pbar.update(1)

        # Close the progress bar when done
        pbar.close()

    # Generate a summary report
    success_count = sum(1 for r in results if r["status"] == "success")
    skipped_count = sum(1 for r in results if r["status"] == "skipped")
    failed_count = sum(1 for r in results if r["status"] == "failed")

    log_info(f"Processed {len(results)} lines:")
    log_info(f"  ✅ Successful: {success_count}")
    log_info(f"  ⏭️ Skipped: {skipped_count}")
    log_info(f"  ❌ Failed: {failed_count}")
    log_info(f"  🔢 Total entries across all files: {total_entries}")

    # Save results to a JSON file
    results_file = os.path.join(OUTPUT_DIR, "processing_results.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            "summary": {
                "total_lines": len(results),
                "successful": success_count,
                "skipped": skipped_count,
                "failed": failed_count,
                "total_entries": total_entries
            },
            "results": results
        }, f, indent=2, ensure_ascii=False)

    log_info(f"Results saved to {results_file}")
    log_success(f"Total entries processed: {total_entries}", "🔢")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process HELM data from download to conversion")
    # group = parser.add_mutually_exclusive_group(required=True)
    parser.add_argument("--csv-file", help="CSV file with 'Run' column containing HELM catalog lines",
                        default=str(DEFAULT_CSV_FILE_PROCESSOR))
    parser.add_argument("--input-line", help="Single line to process")
    parser.add_argument("--adapter-method", help="Filter tasks by Adapter method (e.g., 'multiple_choice_joint')")
    parser.add_argument("--keep-temp", action="store_true", help="Keep temporary files (zip and extracted directories)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")

    args = parser.parse_args()

    log_info(f"Starting HELM Data Processor", "🚀")
    log_info(f"Arguments: {args}", "🔧")

    if args.csv_file:
        log_step(f"Processing from CSV file: {args.csv_file}", "📋")
        main(csv_file=args.csv_file, adapter_method=args.adapter_method,
             keep_temp_files=args.keep_temp, overwrite=args.overwrite)
    elif args.input_file:
        log_step(f"Processing from input file: {args.input_file}", "📄")
        main(input_file=args.input_file, keep_temp_files=args.keep_temp, overwrite=args.overwrite)
    elif args.input_line:
        log_step(f"Processing single line: {args.input_line}", "📝")
        main(input_lines=[args.input_line], keep_temp_files=args.keep_temp, overwrite=args.overwrite)

    log_success(f"HELM Data Processor completed", "🏁")
