import sys
import logging
from logging import Formatter, StreamHandler

class ColorFormatter(Formatter):
    """支持ANSI颜色代码的日志格式化器"""
    COLOR_CODES = {
        logging.DEBUG:    '\033[34m',  # 蓝色
        logging.INFO:     '\033[32m',  # 绿色
        logging.WARNING:  '\033[33m',  # 黄色
        logging.ERROR:    '\033[31m',  # 红色
        logging.CRITICAL: '\033[31;1m' # 红色加粗
    }
    RESET_CODE = '\033[0m'

    def __init__(self, fmt=None, datefmt=None, style='%'):
        super().__init__(fmt, datefmt, style)
        # 仅在终端环境下启用颜色
        self.use_color = sys.stdout.isatty()

    def format(self, record):
        # 获取原始日志信息
        message = super().format(record)
        
        if self.use_color:
            # 添加颜色控制代码
            color = self.COLOR_CODES.get(record.levelno, '')
            return f"{color}{message}{self.RESET_CODE}"
        return message

def setup_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # 定义基础格式（不包含颜色代码）
    base_format = (
        '[%(levelname)s|%(filename)s:%(lineno)d] '
        '%(asctime)s.%(msecs)03d >> %(message)s'
    )
    date_format = '%Y-%m-%d %H:%M:%S'

    # 创建彩色格式化器
    formatter = ColorFormatter(
        fmt=base_format,
        datefmt=date_format
    )

    # 配置控制台处理器
    console_handler = StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # 避免重复日志
    if not logger.handlers:
        logger.addHandler(console_handler)
        logger.propagate = False

    return logger

# 使用示例
logger = setup_logger()

if __name__ == "__main__":
    logger.debug("Debugging information")
    logger.info("System operational")
    logger.warning("Resource usage high")
    logger.error("Network connection lost")
    logger.critical("System core dump imminent")