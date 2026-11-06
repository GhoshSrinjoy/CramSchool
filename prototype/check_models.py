from pathlib import Path
import os
import json
from rich.console import Console
from rich.table import Table
from rich.progress import track
from huggingface_hub import scan_cache_dir, try_to_load_from_cache
import shutil
from datetime import datetime

class HuggingFaceCacheInspector:
    def __init__(self):
        self.console = Console()
        
        # Get cache directory
        if os.name == 'nt':  # Windows
            self.cache_dir = os.path.expanduser('~/.cache/huggingface')
        else:  # Linux/Mac
            self.cache_dir = os.path.expanduser('~/.cache/huggingface')
            
        # Additional common locations
        possible_locations = [
            self.cache_dir,
            os.path.join(os.path.expanduser('~'), '.huggingface'),
            os.path.join(os.path.expanduser('~'), 'AppData', 'Local', 'huggingface'),
            'C:\\Users\\{}\\AppData\\Local\\huggingface'.format(os.getlogin()),
            os.environ.get('TRANSFORMERS_CACHE'),
            os.environ.get('HF_HOME')
        ]
        
        # Find first existing cache directory
        for loc in possible_locations:
            if loc and os.path.exists(loc):
                self.cache_dir = loc
                break
                
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)
            
        self.console.print(f"[cyan]Using cache directory: {self.cache_dir}[/cyan]")

    def scan_directory(self, directory):
        """Scan directory for model files"""
        models = []
        if not os.path.exists(directory):
            return models
            
        for item in os.scandir(directory):
            if item.is_dir() and item.name.startswith('models--'):
                model_name = item.name.replace('models--', '').replace('--', '/')
                models.append({
                    'name': model_name,
                    'path': item.path,
                    'size': self.get_directory_size(item.path),
                    'mtime': item.stat().st_mtime,
                    'files': sum(1 for _ in Path(item.path).rglob('*') if _.is_file())
                })
        return models

    def get_directory_size(self, path):
        """Get size of directory in bytes"""
        total = 0
        try:
            with os.scandir(path) as it:
                for entry in it:
                    if entry.is_file():
                        total += entry.stat().st_size
                    elif entry.is_dir():
                        total += self.get_directory_size(entry.path)
        except (PermissionError, FileNotFoundError):
            pass
        return total

    def format_size(self, size_in_bytes):
        """Format size to human readable format"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_in_bytes < 1024.0:
                return f"{size_in_bytes:.2f} {unit}"
            size_in_bytes /= 1024.0

    def display_models(self):
        """Display cached models in a table"""
        # Create table
        table = Table(title="Cached Hugging Face Models")
        table.add_column("Model Name", style="cyan")
        table.add_column("Size", style="green")
        table.add_column("Last Modified", style="yellow")
        table.add_column("Files", style="blue")

        # Scan for models
        self.console.print("[yellow]Scanning for models...[/yellow]")
        models = self.scan_directory(os.path.join(self.cache_dir, 'hub'))
        
        if not models:
            models = self.scan_directory(self.cache_dir)

        total_size = 0
        model_count = 0

        # Add rows for each model
        for model in track(models, description="Processing models..."):
            try:
                total_size += model['size']
                last_modified = datetime.fromtimestamp(model['mtime'])

                table.add_row(
                    model['name'],
                    self.format_size(model['size']),
                    last_modified.strftime("%Y-%m-%d %H:%M"),
                    str(model['files'])
                )
                model_count += 1
            except Exception as e:
                self.console.print(f"[red]Error processing model {model['name']}: {str(e)}[/red]")

        # Print summary and table
        self.console.print(f"\n[bold green]Found {model_count} models[/bold green]")
        self.console.print(f"[bold yellow]Total cache size: {self.format_size(total_size)}[/bold yellow]")
        self.console.print(f"[bold blue]Cache location: {self.cache_dir}[/bold blue]\n")
        
        if model_count > 0:
            self.console.print(table)
        else:
            self.console.print("[yellow]No cached models found.[/yellow]")

    def check_space(self):
        """Check available disk space"""
        try:
            total, used, free = shutil.disk_usage(self.cache_dir)
            self.console.print("\n[bold]Disk Space Information:[/bold]")
            self.console.print(f"Total: {self.format_size(total)}")
            self.console.print(f"Used: {self.format_size(used)}")
            self.console.print(f"Free: {self.format_size(free)}")
        except Exception as e:
            self.console.print(f"[red]Error checking disk space: {str(e)}[/red]")

def main():
    try:
        inspector = HuggingFaceCacheInspector()
        inspector.display_models()
        inspector.check_space()
    except Exception as e:
        console = Console()
        console.print(f"[bold red]Error: {str(e)}[/bold red]")
        console.print_exception()

if __name__ == "__main__":
    main()