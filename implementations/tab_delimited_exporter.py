"""Tab-delimited file exporter implementation."""

from typing import List, Dict
import csv
from interfaces.file_interface import FileExporter


class TabDelimitedExporter(FileExporter):
    """Export results to tab-delimited files."""

    def export_results(self, file_path: str, results: List[Dict[str, str]]) -> None:
        """
        Export results to a tab-delimited file.

        Args:
            file_path: Path to the output file.
            results: List of dictionaries containing result data.

        Raises:
            ValueError: If results list is empty or invalid.
        """
        if not results:
            raise ValueError("Cannot export empty results list.")

        try:
            # Get all unique keys from all dictionaries
            fieldnames = set()
            for result in results:
                fieldnames.update(result.keys())
            fieldnames = sorted(list(fieldnames))

            with open(file_path, 'w', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames, delimiter='\t')
                writer.writeheader()
                writer.writerows(results)

            print(f"Results exported successfully to: {file_path}")

        except Exception as e:
            raise ValueError(f"Error writing file: {str(e)}")
