import os 
import platform
import subprocess

def get_cpu_name():
    system = platform.system()

    if system == "Windows":
        return platform.processor()
    
    elif system == "Linux":
        command = "cat /proc/cpuinfo"
        all_info = subprocess.check_output(command, shell=True).strip().decode()
        for line in all_info.split("\n"):
            if "model name" in line:
                return line.split(": ", 1)[1]
    
    elif system == "Darwin":
        command = ["sysctl", "-n", "machdep.cpu.brand_string"]
        return subprocess.check_output(command, env={**os.environ, 'PATH': os.environ['PATH'] + os.pathsep + '/usr/sbin'}).strip().decode()

    return ""

def get_gpu_name():
    try: 
        gpu_info = subprocess.check_output("nvidia-smi -L", shell=True, stderr=subprocess.DEVNULL).strip().decode()
        return gpu_info.split(':', 1)[1].split('(', 1)[0].strip()
    except subprocess.CalledProcessError:
        return "No GPU found" 

if __name__ == "__main__":
    print(f"CPU: {get_cpu_name()}")
    print(f"GPU: {get_gpu_name()}")
