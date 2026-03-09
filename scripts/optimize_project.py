#!/usr/bin/env python3
"""
量化项目优化工具
用于自动优化量化研究项目的性能、代码质量和功能

作者：王成龙 (Chenglong Wang)
创建时间：2026-03-05
"""

import os
import sys
import argparse
import yaml
import json
import time
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional
import subprocess
import logging
from dataclasses import dataclass, asdict
from enum import Enum

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OptimizationLevel(Enum):
    """优化级别"""
    BASIC = "basic"      # 基础优化
    ADVANCED = "advanced"  # 高级优化
    COMPREHENSIVE = "comprehensive"  # 全面优化

@dataclass
class OptimizationResult:
    """优化结果"""
    success: bool
    message: str
    files_optimized: List[str]
    performance_improvement: Optional[float] = None
    time_taken: Optional[float] = None
    errors: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []

class ProjectOptimizer:
    """项目优化器"""
    
    def __init__(self, project_path: str, config_path: Optional[str] = None):
        self.project_path = Path(project_path).resolve()
        self.config_path = Path(config_path) if config_path else None
        
        # 验证项目路径
        if not self.project_path.exists():
            raise ValueError(f"项目路径不存在: {self.project_path}")
        
        # 加载配置
        self.config = self._load_config()
        
        # 项目结构
        self.structure = {
            'src': self.project_path / 'src',
            'tests': self.project_path / 'tests',
            'docs': self.project_path / 'docs',
            'scripts': self.project_path / 'scripts',
            'data': self.project_path / 'data',
            'models': self.project_path / 'models',
            'logs': self.project_path / 'logs',
        }
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置"""
        default_config = {
            'optimization': {
                'level': 'comprehensive',
                'backup_before_optimize': True,
                'create_git_commit': True,
            },
            'performance': {
                'enable_gpu': True,
                'enable_parallel': True,
                'memory_optimization': True,
            },
            'code_quality': {
                'add_type_hints': True,
                'improve_error_handling': True,
                'add_docstrings': True,
            }
        }
        
        if self.config_path and self.config_path.exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    user_config = yaml.safe_load(f)
                    # 合并配置
                    self._merge_configs(default_config, user_config)
            except Exception as e:
                logger.warning(f"加载用户配置失败: {e}, 使用默认配置")
        
        return default_config
    
    def _merge_configs(self, base: Dict, update: Dict) -> None:
        """递归合并配置"""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_configs(base[key], value)
            else:
                base[key] = value
    
    def optimize(self, level: OptimizationLevel = OptimizationLevel.COMPREHENSIVE) -> OptimizationResult:
        """执行优化"""
        start_time = time.time()
        files_optimized = []
        errors = []
        
        try:
            logger.info(f"开始优化项目: {self.project_path}")
            logger.info(f"优化级别: {level.value}")
            
            # 1. 备份项目
            if self.config['optimization']['backup_before_optimize']:
                self._backup_project()
            
            # 2. 优化项目结构
            logger.info("优化项目结构...")
            self._optimize_project_structure()
            
            # 3. 根据级别执行优化
            if level == OptimizationLevel.BASIC:
                files_optimized.extend(self._basic_optimization())
            elif level == OptimizationLevel.ADVANCED:
                files_optimized.extend(self._basic_optimization())
                files_optimized.extend(self._advanced_optimization())
            else:  # COMPREHENSIVE
                files_optimized.extend(self._basic_optimization())
                files_optimized.extend(self._advanced_optimization())
                files_optimized.extend(self._comprehensive_optimization())
            
            # 4. 创建Git提交
            if self.config['optimization']['create_git_commit']:
                self._create_git_commit(level)
            
            time_taken = time.time() - start_time
            
            result = OptimizationResult(
                success=True,
                message=f"项目优化完成，级别: {level.value}",
                files_optimized=files_optimized,
                time_taken=time_taken,
                errors=errors
            )
            
            logger.info(f"优化完成，耗时: {time_taken:.2f}秒")
            logger.info(f"优化的文件数: {len(files_optimized)}")
            
        except Exception as e:
            errors.append(str(e))
            logger.error(f"优化过程中出错: {e}")
            
            result = OptimizationResult(
                success=False,
                message=f"优化失败: {str(e)}",
                files_optimized=files_optimized,
                time_taken=time.time() - start_time,
                errors=errors
            )
        
        return result
    
    def _backup_project(self) -> None:
        """备份项目"""
        backup_dir = self.project_path.parent / f"{self.project_path.name}_backup"
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        
        logger.info(f"创建项目备份: {backup_dir}")
        shutil.copytree(self.project_path, backup_dir)
    
    def _optimize_project_structure(self) -> None:
        """优化项目结构"""
        logger.info("创建标准项目结构...")
        
        # 创建目录
        for dir_name, dir_path in self.structure.items():
            dir_path.mkdir(exist_ok=True, parents=True)
            logger.debug(f"创建目录: {dir_path}")
        
        # 创建必要的文件
        self._create_essential_files()
    
    def _create_essential_files(self) -> None:
        """创建必要的文件"""
        essential_files = {
            'src/__init__.py': '# 量化项目源代码包\n',
            'tests/__init__.py': '# 测试包\n',
            'tests/conftest.py': '# pytest配置\nimport pytest\n',
            '.env.example': '# 环境变量示例\nDATA_PATH=./data\nMODEL_PATH=./models\n',
            'setup.py': self._generate_setup_py(),
            'pyproject.toml': self._generate_pyproject_toml(),
        }
        
        for file_path, content in essential_files.items():
            full_path = self.project_path / file_path
            if not full_path.exists():
                full_path.parent.mkdir(exist_ok=True, parents=True)
                with open(full_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                logger.debug(f"创建文件: {file_path}")
    
    def _generate_setup_py(self) -> str:
        """生成setup.py文件"""
        return '''#!/usr/bin/env python3
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="quant-research-projects",
    version="2.0.0",
    author="王成龙 (Chenglong Wang)",
    author_email="chenglong.wang@example.com",
    description="量化研究项目套件",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sheldonwangchenglong-jackie/quant_projects",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
    ],
    extras_require={
        "gpu": ["torch>=1.9.0", "cudatoolkit"],
        "dev": ["pytest>=6.0", "black>=21.0", "flake8>=4.0"],
        "docs": ["sphinx>=4.0", "sphinx-rtd-theme>=1.0"],
    },
)
'''
    
    def _generate_pyproject_toml(self) -> str:
        """生成pyproject.toml文件"""
        return '''[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[tool.black]
line-length = 88
target-version = ['py39']

[tool.isort]
profile = "black"
multi_line_output = 3

[tool.mypy]
python_version = "3.9"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = "-v --tb=short"
'''
    
    def _basic_optimization(self) -> List[str]:
        """基础优化"""
        logger.info("执行基础优化...")
        optimized_files = []
        
        # 优化Python文件
        python_files = list(self.project_path.rglob("*.py"))
        
        for py_file in python_files:
            try:
                if self._optimize_python_file(py_file, basic=True):
                    optimized_files.append(str(py_file.relative_to(self.project_path)))
            except Exception as e:
                logger.warning(f"优化文件失败 {py_file}: {e}")
        
        return optimized_files
    
    def _advanced_optimization(self) -> List[str]:
        """高级优化"""
        logger.info("执行高级优化...")
        optimized_files = []
        
        # 1. 添加类型提示
        if self.config['code_quality']['add_type_hints']:
            optimized_files.extend(self._add_type_hints())
        
        # 2. 改进错误处理
        if self.config['code_quality']['improve_error_handling']:
            optimized_files.extend(self._improve_error_handling())
        
        # 3. 性能优化
        if self.config['performance']['enable_gpu']:
            optimized_files.extend(self._add_gpu_support())
        
        return optimized_files
    
    def _comprehensive_optimization(self) -> List[str]:
        """全面优化"""
        logger.info("执行全面优化...")
        optimized_files = []
        
        # 1. 创建配置文件
        config_file = self.project_path / "config" / "optimized_config.yaml"
        config_file.parent.mkdir(exist_ok=True)
        
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
        
        optimized_files.append(str(config_file.relative_to(self.project_path)))
        
        # 2. 创建测试文件
        optimized_files.extend(self._create_test_files())
        
        # 3. 创建文档
        optimized_files.extend(self._create_documentation())
        
        # 4. 创建部署脚本
        optimized_files.extend(self._create_deployment_scripts())
        
        return optimized_files
    
    def _optimize_python_file(self, file_path: Path, basic: bool = True) -> bool:
        """优化单个Python文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            optimized = False
            
            # 基础优化
            if basic:
                # 添加文件头注释
                if not content.startswith('#!/usr/bin/env python3'):
                    header = '''#!/usr/bin/env python3
"""
优化版本
原始文件: {}
优化时间: {}
"""
'''.format(file_path.name, time.strftime("%Y-%m-%d %H:%M:%S"))
                    content = header + content
                    optimized = True
            
            # 保存优化后的文件
            if optimized:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                return True
            
        except Exception as e:
            logger.warning(f"优化文件 {file_path} 失败: {e}")
        
        return False
    
    def _add_type_hints(self) -> List[str]:
        """添加类型提示"""
        # 这里可以集成mypy或pytype
        logger.info("添加类型提示...")
        return []
    
    def _improve_error_handling(self) -> List[str]:
        """改进错误处理"""
        logger.info("改进错误处理...")
        return []
    
    def _add_gpu_support(self) -> List[str]:
        """添加GPU支持"""
        logger.info("添加GPU支持...")
        return []
    
    def _create_test_files(self) -> List[str]:
        """创建测试文件"""
        logger.info("创建测试文件...")
        test_files = []
        
        # 为每个Python文件创建对应的测试文件
        python_files = list(self.project_path.rglob("*.py"))
        
        for py_file in python_files:
            if "test" in py_file.name or py_file.parent.name == "tests":
                continue
            
            # 创建测试文件路径
            rel_path = py_file.relative_to(self.project_path)
            test_path = self.project_path / "tests" / f"test_{rel_path.name}"
            
            test_content = f'''#!/usr/bin/env python3
"""
测试文件: {rel_path}
"""

import pytest
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_file_exists():
    """测试文件是否存在"""
    assert os.path.exists("{rel_path}")

def test_file_not_empty():
    """测试文件非空"""
    with open("{rel_path}", "r", encoding="utf-8") as f:
        content = f.read()
    assert len(content) > 0

# 添加更多测试...
'''
            
            test_path.parent.mkdir(exist_ok=True, parents=True)
            with open(test_path, 'w', encoding='utf-8') as f:
                f.write(test_content)
            
            test_files.append(str(test_path.relative_to(self.project_path)))
        
        return test_files
    
    def _create_documentation(self) -> List[str]:
        """创建文档"""
        logger.info("创建文档...")
        doc_files = []
        
        # 创建API文档模板
        api_doc = self.project_path / "docs" / "api.md"
        api_doc.parent.mkdir(exist_ok=True, parents=True)
        
        api_content = '''# API 文档

## 概述
量化研究项目API文档。

## 模块

### GAT多因子Alpha策略
- `GATMultiFactorModel`: 图注意力网络模型
- `FactorDataGenerator`: 因子数据生成器
- `BacktestEngine`: 回测引擎

### 亚式期权定价
- `AsianOptionPricer`: 期权定价器
- `RiskAnalysis`: 风险分析工具

### 强化学习期货执行
- `FuturesExecutionEnv`: 期货执行环境
- `PPOAgent`: PPO智能体

## 使用示例
```python
# 示例代码
from src.models.gat import GATMultiFactorModel

model = GATMultiFactorModel()
# ...
```
'''
        
        with open(api_doc, 'w', encoding='utf-8') as f:
            f.write(api_content)
        
        doc_files.append(str(api_doc.relative_to(self.project_path)))
        
        return doc_files
    
    def _create_deployment_scripts(self) -> List[str]:
        """创建部署脚本"""
        logger.info("创建部署脚本...")
        script_files = []
        
        # Dockerfile
        dockerfile = self.project_path / "Dockerfile"
        docker_content = '''FROM python:3.9-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .
COPY pyproject.toml .
COPY setup.py .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -e .

# 复制源代码
COPY src/ ./src/
COPY config/ ./config/
COPY scripts/ ./scripts/

# 创建数据目录
RUN mkdir -p /app/data /app/models /app/logs

# 设置环境变量
ENV PYTHONPATH=/app
ENV DATA_PATH=/app/data
ENV MODEL_PATH=/app/models
ENV LOG_PATH=/app/logs

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["python", "-m", "src.main"]
'''
        
        with open(dockerfile, 'w', encoding='utf-8') as f:
            f.write(docker_content)
        
        script_files.append(str(dockerfile.relative_to(self.project_path)))
        
        return script_files
    
    def _create_git_commit(self, level: OptimizationLevel) -> None:
        """创建Git提交"""
        try:
            subprocess.run(
                ["git", "add", "."],
                cwd=self.project_path,
                check=True,
                capture_output=True
            )
            commit_msg = f"chore: apply {level.value} optimization"
            commit_proc = subprocess.run(
                ["git", "commit", "-m", commit_msg],
                cwd=self.project_path,
                check=False,
                capture_output=True,
                text=True
            )
            if commit_proc.returncode == 0:
                logger.info("已创建Git提交: %s", commit_msg)
            else:
                logger.info("未创建新提交（可能没有变更）")
        except Exception as e:
            logger.warning(f"创建Git提交失败: {e}")


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="量化项目优化工具")
    parser.add_argument(
        "--project-path",
        default=".",
        help="项目根目录路径（默认: 当前目录）"
    )
    parser.add_argument(
        "--config-path",
        default=None,
        help="优化配置文件路径（可选）"
    )
    parser.add_argument(
        "--level",
        default="comprehensive",
        choices=[level.value for level in OptimizationLevel],
        help="优化级别：basic/advanced/comprehensive"
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="将优化结果写入JSON文件（可选）"
    )
    return parser.parse_args()


def main() -> int:
    """命令行入口"""
    args = parse_args()
    level = OptimizationLevel(args.level)

    optimizer = ProjectOptimizer(
        project_path=args.project_path,
        config_path=args.config_path
    )
    result = optimizer.optimize(level=level)

    result_dict = asdict(result)
    print(json.dumps(result_dict, ensure_ascii=False, indent=2))

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result_dict, f, ensure_ascii=False, indent=2)
        logger.info("优化结果已写入: %s", output_path)

    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
