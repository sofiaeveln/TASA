import logging
from datetime import datetime
import os
from pathlib import Path
from logging.handlers import RotatingFileHandler

def set_logger(log_dir='./logs/', log_prefix=''):
    """设置日志器"""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger()
    
    if not logger.hasHandlers():
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s %(levelname)-8s %(message)s',
            "%Y-%m-%d %H:%M:%S")
        
        # 控制台日志
        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)
        sh.setFormatter(formatter)
        logger.addHandler(sh)
        
        # 文件日志
        ts = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        fh = RotatingFileHandler(
            f'{log_dir}/{log_prefix}-{ts}.log', 
            maxBytes=10*1024*1024, 
            backupCount=5
        )
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    
    return logger