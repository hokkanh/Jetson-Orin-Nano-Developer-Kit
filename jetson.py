#!/usr/bin/env python3
"""
Jetson Orin MCAP Video Processor

Processes MCAP video files for object/shape detection.
"""

from pathlib import Path
from typing import List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MCAPProcessor:
    """Simple MCAP file processor."""
    
    def __init__(self, data_dir: str = "data/mcap_files", output_dir: str = "output"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def list_mcap_files(self) -> List[Path]:
        """List all MCAP files in data directory."""
        mcap_files = list(self.data_dir.glob("*.mcap"))
        return sorted(mcap_files)
    
    def show_status(self):
        """Show processing status and next steps."""
        mcap_files = self.list_mcap_files()
        
        print("\n" + "="*60)
        print("MCAP PROCESSOR STATUS")
        print("="*60)
        print(f"\nData directory: {self.data_dir.resolve()}")
        print(f"Output directory: {self.output_dir.resolve()}")
        print(f"\nMCAP files found: {len(mcap_files)}")
        
        for f in mcap_files:
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  - {f.name} ({size_mb:.2f} MB)")
        
        if not mcap_files:
            print("  (none yet - add files to data/mcap_files/)")
        
        print("\n" + "="*60 + "\n")


def main():
    """Main entry point."""
    processor = MCAPProcessor()
    processor.show_status()


if __name__ == '__main__':
    main()