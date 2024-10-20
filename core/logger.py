import datetime
import os
from pathlib import Path
import time

class Severity:
    NONE = "None"
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"

class LogType:
    INFO = "Info"
    DEBUG = "Debug"
    WARNING = "Warning"
    ERROR = "Error"

class Colors:
    @staticmethod
    def ok_blue():
        return "\033[94m"

    @staticmethod
    def ok_green():
        return "\033[92m"

    @staticmethod
    def warning():
        return "\033[93m"

    @staticmethod
    def error():
        return "\033[91m"

    @staticmethod
    def normal():
        return "\033[0m"

class Logger:
    def __init__(self, type_name):
        self.type_name = type_name

    def log(self, log_type: LogType, severity: Severity, message: str):
        time = self.get_time()

        formatted_message = (
            f"[{log_type} - {severity} - "
            f"({self.type_name}) - {self.format_time(time)}]: {message}"
        )

        # Print message to console
        if log_type == LogType.INFO:
            print(f"{Colors.ok_blue()}{formatted_message}{Colors.normal()}")
        elif log_type == LogType.DEBUG:
            if os.getenv("LOGGER_DEBUG", "false").lower() == "true":
                print(f"{Colors.ok_green()}{formatted_message}{Colors.normal()}")
            else:
                return
        elif log_type == LogType.WARNING:
            print(f"{Colors.warning()}{formatted_message}{Colors.normal()}")
        elif log_type == LogType.ERROR:
            print(f"{Colors.error()}{formatted_message}{Colors.normal()}")

        # Write to file if WRITE_LOGS is enabled
        if os.getenv("WRITE_LOGS", "false").lower() == "true":
            self.write_log(formatted_message)

    def info(self, message: str):
        self.log(LogType.INFO, Severity.NONE, message)

    def debug(self, message: str):
        self.log(LogType.DEBUG, Severity.NONE, message)

    def warning(self, message: str, severity: Severity):
        self.log(LogType.WARNING, severity, message)

    def error(self, message: str, severity: Severity):
        self.log(LogType.ERROR, severity, message)

    def get_time(self):
        return int(time.time())

    def format_time(self, unix_time):
        local_time = datetime.datetime.fromtimestamp(unix_time)
        return local_time.strftime("%Y-%m-%d %H:%M:%S")

    def write_log(self, message):
        new_line = "\r\n" if os.name == 'nt' else "\n"
        dir_path = Path("out_data")
        dir_path.mkdir(parents=True, exist_ok=True)

        log_file = dir_path / "dolly.log"
        with open(log_file, "a") as f:
            f.write(f"{message}{new_line}")
