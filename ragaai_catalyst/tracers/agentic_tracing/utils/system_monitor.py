import platform
import psutil
import sys
import pkg_resources
import logging
from typing import Dict, List, Optional
from ..data.data_structure import (
    SystemInfo,
    OSInfo,
    EnvironmentInfo,
    Resources,
    CPUResource,
    MemoryResource,
    DiskResource,
    NetworkResource,
    ResourceInfo,
    MemoryInfo,
    DiskInfo,
    NetworkInfo,
)

logger = logging.getLogger(__name__)

class SystemMonitor:
    def __init__(self, trace_id: str):
        self.trace_id = trace_id

    def get_system_info(self) -> SystemInfo:
        # Initialize with None values
        os_info = OSInfo(
            name=None,
            version=None,
            platform=None,
            kernel_version=None,
        )
        env_info = EnvironmentInfo(
            name=None,
            version=None,
            packages=[],
            env_path=None,
            command_to_run=None,
        )

        try:
            # Get OS info
            os_info = OSInfo(
                name=platform.system(),
                version=platform.version(),
                platform=platform.machine(),
                kernel_version=platform.release(),
            )
        except Exception as e:
            logger.warning(f"Failed to get OS info: {str(e)}")
             
        try:
            # Get Python environment info
            installed_packages = [
                f"{pkg.key}=={pkg.version}" for pkg in pkg_resources.working_set
            ]
            env_info = EnvironmentInfo(
                name="Python",
                version=platform.python_version(),
                packages=installed_packages,
                env_path=sys.prefix,
                command_to_run=f"python {sys.argv[0]}",
            )
        except Exception as e:
            logger.warning(f"Failed to get environment info: {str(e)}")
             

        # Always return a valid SystemInfo object
        return SystemInfo(
            id=f"sys_{self.trace_id}",
            os=os_info,
            environment=env_info,
            source_code="",
        )

    def get_resources(self) -> Resources:
        # Initialize with None values
        cpu_info = ResourceInfo(
            name=None,
            cores=None,
            threads=None,
        )
        cpu = CPUResource(info=cpu_info, interval="5s", values=[])

        mem_info = MemoryInfo(
            total=None,
            free=None,
        )
        mem = MemoryResource(info=mem_info, interval="5s", values=[])

        disk_info = DiskInfo(
            total=None,
            free=None,
        )
        disk_resource = DiskResource(
            info=disk_info,
            interval="5s",
            read=[],
            write=[],
        )

        net_info = NetworkInfo(
            upload_speed=None,
            download_speed=None,
        )
        net = NetworkResource(
            info=net_info,
            interval="5s",
            uploads=[],
            downloads=[],
        )

        try:
            # CPU info
            cpu_info = ResourceInfo(
                name=platform.processor(),
                cores=psutil.cpu_count(logical=False),
                threads=psutil.cpu_count(logical=True),
            )
            cpu = CPUResource(info=cpu_info, interval="5s", values=[psutil.cpu_percent()])
        except Exception as e:
            logger.warning(f"Failed to get CPU info: {str(e)}")
             

        try:
            # Memory info
            memory = psutil.virtual_memory()
            mem_info = MemoryInfo(
                total=memory.total / (1024**3),  # Convert to GB
                free=memory.available / (1024**3),
            )
            mem = MemoryResource(info=mem_info, interval="5s", values=[memory.percent])
        except Exception as e:
            logger.warning(f"Failed to get memory info: {str(e)}")
             

        try:
            # Disk info
            disk = psutil.disk_usage("/")
            disk_info = DiskInfo(total=disk.total / (1024**3), free=disk.free / (1024**3))
            disk_io = psutil.disk_io_counters()
            disk_resource = DiskResource(
                info=disk_info,
                interval="5s",
                read=[disk_io.read_bytes / (1024**2)],  # MB
                write=[disk_io.write_bytes / (1024**2)],
            )
        except Exception as e:
            logger.warning(f"Failed to get disk info: {str(e)}")
             
        try:
            # Network info
            net_io = psutil.net_io_counters()
            net_info = NetworkInfo(
                upload_speed=net_io.bytes_sent / (1024**2),  # MB
                download_speed=net_io.bytes_recv / (1024**2),
            )
            net = NetworkResource(
                info=net_info,
                interval="5s",
                uploads=[net_io.bytes_sent / (1024**2)],
                downloads=[net_io.bytes_recv / (1024**2)],
            )
        except Exception as e:
            logger.warning(f"Failed to get network info: {str(e)}")
            

        # Always return a valid Resources object
        return Resources(cpu=cpu, memory=mem, disk=disk_resource, network=net)

    def track_memory_usage(self) -> Optional[float]:
        """Track memory usage in MB"""
        try:
            memory_usage = psutil.Process().memory_info().rss
            return memory_usage / (1024 * 1024)  # Convert to MB
        except Exception as e:
            logger.warning(f"Failed to track memory usage: {str(e)}")
            return None  

    def track_cpu_usage(self, interval: float) -> Optional[float]:
        """Track CPU usage percentage"""
        try:
            return psutil.cpu_percent(interval=interval)
        except Exception as e:
            logger.warning(f"Failed to track CPU usage: {str(e)}")
            return None  

    def track_disk_usage(self) -> Dict[str, Optional[float]]:
        """Track disk I/O in MB"""
        default_response = {'disk_read': None, 'disk_write': None}
        try:
            disk_io = psutil.disk_io_counters()
            return {
                'disk_read': disk_io.read_bytes / (1024 * 1024),  # Convert to MB
                'disk_write': disk_io.write_bytes / (1024 * 1024)  # Convert to MB
            }
        except Exception as e:
            logger.warning(f"Failed to track disk usage: {str(e)}")
            return default_response 

    def track_network_usage(self) -> Dict[str, Optional[float]]:
        """Track network I/O in MB"""
        default_response = {'uploads': None, 'downloads': None}
        try:
            net_io = psutil.net_io_counters()
            return {
                'uploads': net_io.bytes_sent / (1024 * 1024),  # Convert to MB
                'downloads': net_io.bytes_recv / (1024 * 1024)  # Convert to MB
            }
        except Exception as e:
            logger.warning(f"Failed to track network usage: {str(e)}")
            return default_response 
