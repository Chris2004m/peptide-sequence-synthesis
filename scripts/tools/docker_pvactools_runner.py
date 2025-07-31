#!/usr/bin/env python3
"""
Docker PVACtools Runner with Griffith Lab Official Image
Addresses common Docker issues: volume mounting, permissions, output accessibility
"""

import os
import subprocess
import sys
from pathlib import Path
import shutil

class DockerPVACRunner:
    def __init__(self, project_root):
        self.project_root = Path(project_root).resolve()
        self.docker_image = "griffithlab/pvactools:latest"
        self.data_dir = self.project_root / "data"
        self.results_dir = self.project_root / "results" / "pvacbind_docker"
        
        # All available prediction algorithms in PVACtools
        self.all_algorithms = [
            "MHCflurry",
            "MHCnuggetsI", 
            "MHCnuggetsII",
            "NetMHC",
            "NetMHCcons",
            "NetMHCpan",
            "NetMHCpanEL",
            "NetMHCII",
            "NetMHCIIpan",
            "PickPocket",
            "SMM",
            "SMMPMBEC",
            "SMMalign"
        ]
        
        # Input files for the three peptide sets
        self.input_files = {
            "random": "random_9mers_10k.fasta",
            "fasta": "fasta_9mers_10k.fasta", 
            "llm": "llm_9mers_10k.fasta"
        }
        
    def check_docker(self):
        """Check if Docker is available and running"""
        try:
            result = subprocess.run(["docker", "--version"], 
                                  capture_output=True, text=True, check=True)
            print(f"✓ Docker found: {result.stdout.strip()}")
            
            # Check if Docker daemon is running
            result = subprocess.run(["docker", "info"], 
                                  capture_output=True, text=True, check=True)
            print("✓ Docker daemon is running")
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ Docker issue: {e}")
            return False
            
    def pull_docker_image(self):
        """Pull the latest Griffith Lab PVACtools image"""
        print(f"Pulling Docker image: {self.docker_image}")
        try:
            subprocess.run([
                "docker", "pull", self.docker_image
            ], check=True)
            print("✓ Docker image pulled successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to pull image: {e}")
            return False
            
    def setup_directories(self):
        """Create necessary directories with proper permissions"""
        directories = [
            self.results_dir,
            self.results_dir / "random",
            self.results_dir / "fasta", 
            self.results_dir / "llm"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            # Set permissive permissions for Docker volume mounting
            os.chmod(directory, 0o777)
            print(f"✓ Created directory: {directory}")
            
    def check_input_files(self):
        """Verify all input files exist"""
        missing_files = []
        for name, filename in self.input_files.items():
            filepath = self.data_dir / filename
            if not filepath.exists():
                missing_files.append(f"{name}: {filepath}")
            else:
                print(f"✓ Found input file: {filepath}")
                
        if missing_files:
            print("✗ Missing input files:")
            for file in missing_files:
                print(f"  - {file}")
            return False
        return True
        
    def run_pvacbind_docker(self, sample_name, input_file):
        """Run PVACbind using Docker with comprehensive algorithm set"""
        input_path = self.data_dir / input_file
        output_dir = self.results_dir / sample_name
        
        # Docker volume mounts
        volumes = [
            f"{self.data_dir}:/data:ro",  # Read-only data mount
            f"{output_dir}:/output:rw"    # Read-write output mount
        ]
        
        # PVACbind command with all algorithms
        cmd = [
            "docker", "run", "--rm",
            "-u", f"{os.getuid()}:{os.getgid()}",  # Run as current user
            "--platform", "linux/amd64",  # Specify platform for M1 Macs
        ]
        
        # Add volume mounts
        for volume in volumes:
            cmd.extend(["-v", volume])
            
        cmd.extend([
            self.docker_image,
            "pvacbind", "run",
            f"/data/{input_file}",
            sample_name,
            "HLA-A*02:01",  # Primary allele
            ",".join(self.all_algorithms),  # All algorithms
            "/output",
            "-e", "9",  # 9-mer peptides
            "-b", "500",  # Binding threshold
            "--iedb-install-directory", "/opt/iedb",
            "--additional-report-columns", "sample_name",
            "--keep-tmp-files",
            "-t", "4"  # Use 4 threads
        ])
        
        print(f"\n🐳 Running PVACbind for {sample_name}")
        print(f"Command: {' '.join(cmd)}")
        print(f"Input: {input_path}")
        print(f"Output: {output_dir}")
        print(f"Algorithms: {', '.join(self.all_algorithms)}")
        
        try:
            # Run with real-time output
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Print output in real-time
            for line in process.stdout:
                print(f"[{sample_name}] {line.rstrip()}")
                
            process.wait()
            
            if process.returncode == 0:
                print(f"✓ {sample_name} completed successfully")
                return True
            else:
                print(f"✗ {sample_name} failed with return code {process.returncode}")
                return False
                
        except Exception as e:
            print(f"✗ Error running {sample_name}: {e}")
            return False
            
    def verify_outputs(self):
        """Verify that outputs were generated correctly"""
        success_count = 0
        
        for sample_name in self.input_files.keys():
            output_dir = self.results_dir / sample_name / "MHC_Class_I"
            epitopes_file = output_dir / f"{sample_name}.all_epitopes.tsv"
            
            if epitopes_file.exists():
                file_size = epitopes_file.stat().st_size
                print(f"✓ {sample_name}: {epitopes_file} ({file_size:,} bytes)")
                success_count += 1
            else:
                print(f"✗ {sample_name}: Missing {epitopes_file}")
                
        return success_count == len(self.input_files)
        
    def run_all(self):
        """Run complete Docker PVACtools workflow"""
        print("🧬 Starting Docker PVACtools Analysis")
        print("=" * 50)
        
        # Pre-flight checks
        if not self.check_docker():
            return False
            
        if not self.pull_docker_image():
            return False
            
        if not self.check_input_files():
            return False
            
        self.setup_directories()
        
        # Run analysis for each peptide set
        success_count = 0
        for sample_name, input_file in self.input_files.items():
            if self.run_pvacbind_docker(sample_name, input_file):
                success_count += 1
            print()  # Add spacing between runs
            
        # Verify results
        print("📊 Verifying Results")
        print("-" * 30)
        if self.verify_outputs():
            print(f"✅ All {len(self.input_files)} analyses completed successfully!")
            print(f"Results available in: {self.results_dir}")
        else:
            print(f"⚠️  Only {success_count}/{len(self.input_files)} analyses completed")
            
        return success_count == len(self.input_files)


def main():
    # Get project root directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    
    print(f"Project root: {project_root}")
    
    runner = DockerPVACRunner(project_root)
    success = runner.run_all()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
