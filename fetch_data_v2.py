"""
台股資料抓取與 Qlib 格式轉換 - 15分鐘頻率版
包含更新、驗證、視覺化等功能，專為分鐘級資料優化
"""

import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import argparse
import json


class TaiwanStockQlibConverter:
    """台股 Qlib 資料轉換器 - 15分鐘版"""
    
    def __init__(self, stocks_dict, csv_dir="./tw_stock_data/csv", 
                 qlib_dir="./tw_stock_data/qlib_data"):
        """
        初始化
        
        Parameters:
        -----------
        stocks_dict : dict
            股票代碼映射
        csv_dir : str
            CSV 輸出目錄
        qlib_dir : str
            Qlib 格式輸出目錄
        """
        self.stocks = stocks_dict
        self.csv_dir = Path(csv_dir)
        self.qlib_dir = Path(qlib_dir)
        self.csv_dir.mkdir(parents=True, exist_ok=True)
        self.qlib_dir.mkdir(parents=True, exist_ok=True)
        
    def download_stock(self, symbol, yahoo_symbol, period='7d', 
                       interval='15m', retry=3):
        """
        下載單支股票資料 (支援重試) - 使用 period 參數
        
        Parameters:
        -----------
        period : str
            時間週期: '1d', '5d', '7d', '1mo', '3mo', '6mo', '1y', '2y', '5y', 'max'
            注意: yfinance 的分鐘資料有限制，通常最多 60 天
        interval : str
            時間間隔: '1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo'
        retry : int
            重試次數
        """
        for attempt in range(retry):
            try:
                print(f"下載 {symbol} ({yahoo_symbol}) - 嘗試 {attempt + 1}/{retry}")
                print(f"  週期: {period}, 間隔: {interval}")
                
                # 使用 yf.download 取得資料
                df = yf.download(
                    tickers=yahoo_symbol,
                    period=period,
                    interval=interval,
                    auto_adjust=False,
                    progress=False
                )
                
                if df.empty:
                    print(f"  警告: {symbol} 沒有資料")
                    continue
                
                # 清除時區資訊
                if df.index.tz is not None:
                    df.index = df.index.tz_convert(None)
                
                # 處理欄位
                df = df.rename(columns={
                    'Open': 'open',
                    'High': 'high',
                    'Low': 'low',
                    'Close': 'close',
                    'Volume': 'volume',
                    'Adj Close': 'adjclose'
                })
                
                # 計算復權因子
                df['factor'] = df['adjclose'] / df['close']
                df['factor'] = df['factor'].replace([np.inf, -np.inf], 1.0)
                df['factor'] = df['factor'].fillna(1.0)
                
                # 重設索引
                df = df.reset_index()
                # 統一使用 datetime 欄位名稱
                if 'Date' in df.columns:
                    df = df.rename(columns={'Date': 'datetime'})
                elif 'Datetime' in df.columns:
                    df = df.rename(columns={'Datetime': 'datetime'})
                
                df['symbol'] = symbol
                
                # 選取欄位
                columns = ['datetime', 'symbol', 'open', 'close', 'high', 'low', 'volume', 'factor']
                df = df[columns]
                
                # 處理暫停交易（成交量為 0 的情況）
                mask = df['volume'] == 0
                df.loc[mask, ['open', 'close', 'high', 'low', 'volume', 'factor']] = np.nan
                
                print(f"  ✓ 成功下載 {len(df)} 筆資料")
                print(f"  時間範圍: {df['datetime'].min()} ~ {df['datetime'].max()}")
                return df
                
            except Exception as e:
                print(f"  ✗ 嘗試 {attempt + 1} 失敗: {str(e)}")
                if attempt < retry - 1:
                    print(f"  等待 2 秒後重試...")
                    import time
                    time.sleep(2)
        
        return None
    
    def download_all(self, period='7d', interval='15m', save_individual=True):
        """
        下載所有股票資料
        
        Parameters:
        -----------
        period : str
            時間週期
        interval : str
            時間間隔
        save_individual : bool
            是否儲存個別股票的 CSV
        """
        print("=" * 60)
        print(f"開始下載 {len(self.stocks)} 支股票資料")
        print(f"週期: {period}")
        print(f"間隔: {interval}")
        print("=" * 60)
        
        all_data = []
        success_count = 0
        
        for symbol, yahoo_symbol in self.stocks.items():
            df = self.download_stock(symbol, yahoo_symbol, period, interval)
            
            if df is not None:
                # 儲存個別 CSV
                if save_individual:
                    csv_file = self.csv_dir / f"{symbol}_{interval}.csv"
                    df.to_csv(csv_file, index=False, encoding='utf-8-sig')
                    print(f"  儲存至 {csv_file}")
                
                all_data.append(df)
                success_count += 1
            
            print()
        
        # 合併所有資料
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            combined_df = combined_df.sort_values(['datetime', 'symbol'])
            combined_file = self.csv_dir / f"all_stocks_{interval}.csv"
            combined_df.to_csv(combined_file, index=False, encoding='utf-8-sig')
            
            print("=" * 60)
            print(f"✓ 下載完成!")
            print(f"  成功: {success_count}/{len(self.stocks)}")
            print(f"  總資料筆數: {len(combined_df)}")
            print(f"  合併檔案: {combined_file}")
            print("=" * 60)
            
            return combined_df
        else:
            print("✗ 沒有成功下載任何資料")
            return None
    
    def validate_data(self, interval='15m'):
        """驗證資料品質"""
        print("\n資料品質檢查:")
        print("=" * 60)
        
        csv_files = list(self.csv_dir.glob(f"*_{interval}.csv"))
        if not csv_files:
            print(f"找不到 {interval} 間隔的 CSV 檔案")
            return
        
        for csv_file in csv_files:
            if csv_file.name.startswith("all_stocks"):
                continue
                
            print(f"\n檢查 {csv_file.name}:")
            df = pd.read_csv(csv_file)
            
            # 基本統計
            print(f"  資料筆數: {len(df)}")
            print(f"  時間範圍: {df['datetime'].min()} ~ {df['datetime'].max()}")
            
            # 缺失值
            missing = df.isnull().sum()
            if missing.sum() > 0:
                print(f"  ⚠ 缺失值:")
                for col, count in missing[missing > 0].items():
                    print(f"    {col}: {count} ({count/len(df)*100:.2f}%)")
            else:
                print(f"  ✓ 無缺失值")
            
            # Factor 檢查
            factor_stats = df['factor'].describe()
            if factor_stats['min'] < 0.5 or factor_stats['max'] > 2.0:
                print(f"  ⚠ Factor 異常:")
                print(f"    範圍: {factor_stats['min']:.4f} ~ {factor_stats['max']:.4f}")
            else:
                print(f"  ✓ Factor 正常 ({factor_stats['mean']:.4f})")
            
            # 價格異常檢查
            if (df['close'] <= 0).any():
                print(f"  ⚠ 發現負數或零價格")
            else:
                print(f"  ✓ 價格正常")
            
            # 時間連續性檢查（針對分鐘資料）
            df['datetime'] = pd.to_datetime(df['datetime'])
            time_diff = df['datetime'].diff()
            expected_interval = self._get_timedelta(interval)
            
            # 檢查非交易時間外的缺口
            irregular_gaps = time_diff[time_diff > expected_interval * 2]
            if len(irregular_gaps) > 0:
                print(f"  ⚠ 發現 {len(irregular_gaps)} 個時間缺口")
            else:
                print(f"  ✓ 時間序列連續")
    
    def _get_timedelta(self, interval):
        """轉換間隔字串為 timedelta"""
        interval_map = {
            '1m': timedelta(minutes=1),
            '2m': timedelta(minutes=2),
            '5m': timedelta(minutes=5),
            '15m': timedelta(minutes=15),
            '30m': timedelta(minutes=30),
            '60m': timedelta(hours=1),
            '1h': timedelta(hours=1),
            '1d': timedelta(days=1),
        }
        return interval_map.get(interval, timedelta(minutes=15))
    
    def update_data(self, period='7d', interval='15m'):
        """
        增量更新資料
        
        Parameters:
        -----------
        period : str
            更新週期
        interval : str
            資料間隔
        """
        print(f"增量更新資料 (週期: {period}, 間隔: {interval})")
        
        # 讀取現有資料
        for symbol, yahoo_symbol in self.stocks.items():
            csv_file = self.csv_dir / f"{symbol}_{interval}.csv"
            
            if csv_file.exists():
                existing_df = pd.read_csv(csv_file)
                latest_datetime = existing_df['datetime'].max()
                print(f"{symbol}: 最新資料時間 {latest_datetime}")
            else:
                print(f"{symbol}: 無現有資料，進行完整下載")
                existing_df = None
            
            # 下載新資料
            new_df = self.download_stock(symbol, yahoo_symbol, period, interval)
            
            if new_df is not None:
                if existing_df is not None:
                    # 合併資料，去除重複
                    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                    combined_df = combined_df.drop_duplicates(subset=['datetime'], keep='last')
                    combined_df = combined_df.sort_values('datetime')
                else:
                    combined_df = new_df
                
                # 儲存
                combined_df.to_csv(csv_file, index=False, encoding='utf-8-sig')
                print(f"  更新完成: {len(combined_df)} 筆資料")
    
    def export_stock_list(self):
        """匯出股票清單 (for Qlib instruments)"""
        instruments_file = self.csv_dir / "instruments.txt"
        
        with open(instruments_file, 'w') as f:
            for symbol in self.stocks.keys():
                f.write(f"{symbol}\n")
        
        print(f"股票清單已匯出至 {instruments_file}")
    
    def generate_summary(self, interval='15m'):
        """生成資料摘要報告"""
        summary = {
            "股票數量": len(self.stocks),
            "股票列表": list(self.stocks.keys()),
            "資料間隔": interval,
            "CSV 目錄": str(self.csv_dir),
            "Qlib 目錄": str(self.qlib_dir),
            "資料統計": {}
        }
        
        # 統計每支股票的資料
        for symbol in self.stocks.keys():
            csv_file = self.csv_dir / f"{symbol}_{interval}.csv"
            if csv_file.exists():
                df = pd.read_csv(csv_file)
                summary["資料統計"][symbol] = {
                    "筆數": len(df),
                    "起始時間": df['datetime'].min(),
                    "結束時間": df['datetime'].max(),
                    "缺失率": f"{df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100:.2f}%"
                }
        
        # 儲存 JSON
        summary_file = self.csv_dir / f"summary_{interval}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print(f"摘要報告已儲存至 {summary_file}")
        return summary


def main():
    """主程式"""
    parser = argparse.ArgumentParser(description='台股資料轉換為 Qlib 格式 - 15分鐘版')
    parser.add_argument('--action', type=str, default='download',
                       choices=['download', 'update', 'validate', 'summary'],
                       help='執行動作: download(下載), update(更新), validate(驗證), summary(摘要)')
    parser.add_argument('--period', type=str, default='7d',
                       help='時間週期 (1d, 5d, 7d, 1mo, 3mo, 6mo, 1y, 2y, max)')
    parser.add_argument('--interval', type=str, default='15m',
                       choices=['1m', '2m', '5m', '15m', '30m', '60m', '1h', '1d'],
                       help='資料間隔 (預設 15m)')
    
    args = parser.parse_args()
    
    # 台股代碼
    STOCKS = {
        "2330": "2330.TW",      # 台積電
        "2303": "2303.TW",      # 聯電
        "2454": "2454.TW",      # 聯發科
        "3711": "3711.TW",      # 日月光投控
        "3037": "3037.TW",      # 欣興
        "2379": "2379.TW",      # 瑞昱
        "5347": "5347.TWO",     # 世界先進
        "2408": "2408.TW",      # 南亞科
        "006208": "006208.TW",  # 國泰台灣5G+
    }
    
    # 建立轉換器
    converter = TaiwanStockQlibConverter(STOCKS)
    
    # 執行動作
    if args.action == 'download':
        converter.download_all(args.period, args.interval)
        converter.export_stock_list()
        converter.generate_summary(args.interval)
        
    elif args.action == 'update':
        converter.update_data(args.period, args.interval)
        converter.generate_summary(args.interval)
        
    elif args.action == 'validate':
        converter.validate_data(args.interval)
        
    elif args.action == 'summary':
        summary = converter.generate_summary(args.interval)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    
    print("\n完成!")


if __name__ == "__main__":
    main()

