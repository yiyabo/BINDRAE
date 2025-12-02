#!/usr/bin/env python3
"""
下载 AHoJ-DB 所需的 PDB 结构文件

这个脚本会：
1. 扫描 AHoJ-DB 的所有条目
2. 提取所有需要的 PDB ID（包括 query、apo、holo）
3. 从 RCSB PDB 下载结构文件
4. 保存到 data/ahojdb_v2c/pdb_files/ 目录

用法：
    python scripts/download_ahojdb_pdbs.py --data-dir data/ahojdb_v2c --workers 8
    
    # 只统计需要下载的数量（不下载）
    python scripts/download_ahojdb_pdbs.py --data-dir data/ahojdb_v2c --dry-run
    
    # 只下载前 N 个条目涉及的 PDB（用于测试）
    python scripts/download_ahojdb_pdbs.py --data-dir data/ahojdb_v2c --limit 100
"""

import os
import sys
import json
import csv
import gzip
import shutil
import argparse
import requests
from pathlib import Path
from typing import Set, List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PDBDownloader:
    """PDB 结构文件下载器"""
    
    # RCSB PDB 下载 URL 模板
    RCSB_URL_TEMPLATES = [
        "https://files.rcsb.org/download/{pdb_id}.pdb.gz",
        "https://files.rcsb.org/download/{pdb_id}.pdb",
        "https://files.rcsb.org/download/{pdb_id}.cif.gz",  # 备用 mmCIF 格式
    ]
    
    def __init__(
        self,
        data_dir: str,
        output_dir: Optional[str] = None,
        workers: int = 8,
        timeout: int = 30,
        retry: int = 3,
    ):
        """
        Args:
            data_dir: AHoJ-DB 数据目录 (包含 data/ 子目录)
            output_dir: PDB 文件输出目录，默认为 data_dir/pdb_files
            workers: 并行下载线程数
            timeout: 请求超时时间（秒）
            retry: 失败重试次数
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir) if output_dir else self.data_dir / "pdb_files"
        self.workers = workers
        self.timeout = timeout
        self.retry = retry
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 统计信息
        self.stats = {
            "total_entries": 0,
            "unique_pdbs": 0,
            "downloaded": 0,
            "skipped": 0,
            "failed": 0,
        }
        
    def collect_pdb_ids_from_entries(self, limit: Optional[int] = None) -> Set[str]:
        """
        从 AHoJ-DB 条目中收集所有需要的 PDB ID
        
        Args:
            limit: 只处理前 N 个条目（用于测试）
            
        Returns:
            所有需要的 PDB ID 集合
        """
        pdb_ids = set()
        entries_data_dir = self.data_dir / "data"
        
        if not entries_data_dir.exists():
            raise FileNotFoundError(f"数据目录不存在: {entries_data_dir}")
        
        # 获取所有分片目录
        shard_dirs = sorted([d for d in entries_data_dir.iterdir() if d.is_dir()])
        logger.info(f"找到 {len(shard_dirs)} 个分片目录")
        
        entry_count = 0
        
        for shard_dir in tqdm(shard_dirs, desc="扫描分片目录"):
            # 获取该分片下的所有条目
            entry_dirs = [d for d in shard_dir.iterdir() if d.is_dir()]
            
            for entry_dir in entry_dirs:
                if limit and entry_count >= limit:
                    break
                    
                entry_count += 1
                
                # 1. 从条目名称提取 query PDB ID
                # 格式: {pdb_id}-{chain}-{ligand}-{residue}
                entry_name = entry_dir.name
                parts = entry_name.split("-")
                if len(parts) >= 1:
                    query_pdb = parts[0].lower()
                    pdb_ids.add(query_pdb)
                
                # 2. 从 apo_filtered_sorted_results.csv 提取 APO PDB IDs
                apo_csv = entry_dir / "apo_filtered_sorted_results.csv"
                if apo_csv.exists():
                    pdb_ids.update(self._extract_pdb_ids_from_csv(apo_csv))
                
                # 3. 从 holo_filtered_sorted_results.csv 提取 HOLO PDB IDs
                holo_csv = entry_dir / "holo_filtered_sorted_results.csv"
                if holo_csv.exists():
                    pdb_ids.update(self._extract_pdb_ids_from_csv(holo_csv))
            
            if limit and entry_count >= limit:
                break
        
        self.stats["total_entries"] = entry_count
        self.stats["unique_pdbs"] = len(pdb_ids)
        
        logger.info(f"扫描了 {entry_count} 个条目")
        logger.info(f"找到 {len(pdb_ids)} 个唯一 PDB ID")
        
        return pdb_ids
    
    def collect_pdb_ids_from_entries_csv(self, limit: Optional[int] = None) -> Set[str]:
        """
        从 entries.csv 快速收集 PDB ID（不遍历目录）
        
        这个方法更快，但可能不完整（只有 query PDB）
        """
        pdb_ids = set()
        entries_csv = self.data_dir / "entries.csv"
        
        if not entries_csv.exists():
            logger.warning(f"entries.csv 不存在，使用目录扫描方式")
            return self.collect_pdb_ids_from_entries(limit)
        
        with open(entries_csv, 'r') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if limit and i >= limit:
                    break
                # 假设有 pdb_id 或 structure 列
                for col in ['pdb_id', 'structure', 'query_pdb', 'pdb']:
                    if col in row and row[col]:
                        pdb_ids.add(row[col].lower()[:4])
                        break
        
        return pdb_ids
    
    def _extract_pdb_ids_from_csv(self, csv_path: Path) -> Set[str]:
        """从 apo/holo CSV 文件中提取 PDB IDs"""
        pdb_ids = set()
        try:
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # structure 列包含 PDB ID
                    if 'structure' in row and row['structure']:
                        pdb_id = row['structure'].lower()[:4]
                        pdb_ids.add(pdb_id)
        except Exception as e:
            logger.warning(f"读取 {csv_path} 失败: {e}")
        return pdb_ids
    
    def download_pdb(self, pdb_id: str) -> bool:
        """
        下载单个 PDB 文件
        
        Args:
            pdb_id: 4字符 PDB ID
            
        Returns:
            是否下载成功
        """
        pdb_id = pdb_id.lower()
        output_path = self.output_dir / f"{pdb_id}.pdb"
        
        # 如果已存在，跳过
        if output_path.exists():
            return True
        
        # 尝试多个 URL 模板
        for url_template in self.RCSB_URL_TEMPLATES:
            url = url_template.format(pdb_id=pdb_id)
            
            for attempt in range(self.retry):
                try:
                    response = requests.get(url, timeout=self.timeout)
                    
                    if response.status_code == 200:
                        content = response.content
                        
                        # 如果是 gzip 压缩的，解压
                        if url.endswith('.gz'):
                            import io
                            content = gzip.decompress(content)
                        
                        # 保存文件
                        with open(output_path, 'wb') as f:
                            f.write(content)
                        
                        return True
                    
                    elif response.status_code == 404:
                        # 文件不存在，尝试下一个 URL 模板
                        break
                    
                except requests.exceptions.Timeout:
                    logger.warning(f"下载 {pdb_id} 超时 (尝试 {attempt + 1}/{self.retry})")
                    time.sleep(1)
                except Exception as e:
                    logger.warning(f"下载 {pdb_id} 失败: {e} (尝试 {attempt + 1}/{self.retry})")
                    time.sleep(1)
        
        return False
    
    def download_all(
        self, 
        pdb_ids: Set[str], 
        dry_run: bool = False
    ) -> Dict[str, List[str]]:
        """
        并行下载所有 PDB 文件
        
        Args:
            pdb_ids: 要下载的 PDB ID 集合
            dry_run: 如果为 True，只统计不下载
            
        Returns:
            包含成功/失败列表的字典
        """
        # 过滤已存在的文件
        to_download = []
        skipped = []
        
        for pdb_id in pdb_ids:
            output_path = self.output_dir / f"{pdb_id.lower()}.pdb"
            if output_path.exists():
                skipped.append(pdb_id)
            else:
                to_download.append(pdb_id)
        
        logger.info(f"需要下载: {len(to_download)} 个")
        logger.info(f"已存在跳过: {len(skipped)} 个")
        
        self.stats["skipped"] = len(skipped)
        
        if dry_run:
            logger.info("Dry run 模式，不执行下载")
            return {"to_download": to_download, "skipped": skipped, "failed": []}
        
        # 并行下载
        downloaded = []
        failed = []
        
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            future_to_pdb = {
                executor.submit(self.download_pdb, pdb_id): pdb_id 
                for pdb_id in to_download
            }
            
            for future in tqdm(
                as_completed(future_to_pdb), 
                total=len(to_download),
                desc="下载 PDB 文件"
            ):
                pdb_id = future_to_pdb[future]
                try:
                    if future.result():
                        downloaded.append(pdb_id)
                    else:
                        failed.append(pdb_id)
                except Exception as e:
                    logger.error(f"下载 {pdb_id} 异常: {e}")
                    failed.append(pdb_id)
        
        self.stats["downloaded"] = len(downloaded)
        self.stats["failed"] = len(failed)
        
        logger.info(f"下载完成: {len(downloaded)} 个")
        logger.info(f"下载失败: {len(failed)} 个")
        
        # 保存失败列表
        if failed:
            failed_path = self.output_dir / "failed_downloads.txt"
            with open(failed_path, 'w') as f:
                f.write('\n'.join(failed))
            logger.info(f"失败列表保存到: {failed_path}")
        
        return {
            "downloaded": downloaded,
            "skipped": skipped,
            "failed": failed,
        }
    
    def print_stats(self):
        """打印统计信息"""
        print("\n" + "=" * 50)
        print("下载统计")
        print("=" * 50)
        print(f"扫描条目数: {self.stats['total_entries']}")
        print(f"唯一 PDB 数: {self.stats['unique_pdbs']}")
        print(f"已存在跳过: {self.stats['skipped']}")
        print(f"本次下载: {self.stats['downloaded']}")
        print(f"下载失败: {self.stats['failed']}")
        print("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="下载 AHoJ-DB 所需的 PDB 结构文件"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="AHoJ-DB 数据目录路径 (包含 data/ 子目录)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="PDB 文件输出目录，默认为 data-dir/pdb_files"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="并行下载线程数 (默认: 8)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="只处理前 N 个条目 (用于测试)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只统计需要下载的数量，不执行下载"
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="使用 entries.csv 快速收集 (只收集 query PDB)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="请求超时时间（秒）"
    )
    parser.add_argument(
        "--retry",
        type=int,
        default=3,
        help="失败重试次数"
    )
    
    args = parser.parse_args()
    
    # 创建下载器
    downloader = PDBDownloader(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        workers=args.workers,
        timeout=args.timeout,
        retry=args.retry,
    )
    
    # 收集 PDB IDs
    logger.info("开始收集需要下载的 PDB IDs...")
    
    if args.fast:
        pdb_ids = downloader.collect_pdb_ids_from_entries_csv(limit=args.limit)
    else:
        pdb_ids = downloader.collect_pdb_ids_from_entries(limit=args.limit)
    
    # 下载
    logger.info("开始下载...")
    results = downloader.download_all(pdb_ids, dry_run=args.dry_run)
    
    # 打印统计
    downloader.print_stats()
    
    # 如果有失败的，返回非零退出码
    if results.get("failed"):
        sys.exit(1)


if __name__ == "__main__":
    main()
