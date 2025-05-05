import duckdb
import timeit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class DuckDBJoinBenchmark:
    def __init__(self, data_size=1e6):
        self.data_size = int(data_size)
        self.conn = duckdb.connect(':memory:')
        self._load_data()
        
    def _load_data(self):
        # Generate synthetic lineitem and part tables
        lineitem = pd.DataFrame({
            'orderkey': np.arange(self.data_size),
            'totalprice': np.random.normal(100, 50, self.data_size),
            'partkey': np.random.randint(1, 2000, self.data_size)
        })
        
        unique_partkeys = np.unique(lineitem['partkey'])
        part = pd.DataFrame({
            'partkey': unique_partkeys,
            'name': pd.Categorical([f'Part_{i}' for i in range(len(unique_partkeys))]),
            'retailprice': np.random.uniform(10, 500, len(unique_partkeys))
        })
        
        self.conn.register('lineitem', lineitem)
        self.conn.register('part', part)
    
    def _benchmark(self, query, iterations=10):
        times = []
        for _ in range(2):
            self.conn.execute(query)
        for _ in range(iterations):
            start = timeit.default_timer()
            self.conn.execute(query).fetchall()
            times.append(timeit.default_timer() - start)
        return {
            'mean': np.mean(times),
            'std': np.std(times)
        }
    
    def run_analysis(self):
        join_query = """
            SELECT p.name, SUM(l.totalprice)
            FROM lineitem l
            JOIN part p ON l.partkey = p.partkey
            GROUP BY p.name
        """
        return {'Join': self._benchmark(join_query)}
    
    def visualize(self, results, save_path='join_benchmark_results.png'):
        fig, ax = plt.subplots(figsize=(8, 5))
        metrics = list(results.keys())
        means = [results[m]['mean'] for m in metrics]
        stds = [results[m]['std'] for m in metrics]
        
        ax.bar(metrics, means, yerr=stds, capsize=8)
        ax.set_title(f'DuckDB Join Benchmark ({self.data_size:,} rows)')
        ax.set_ylabel('Execution Time (seconds)')
        ax.grid(axis='y', linestyle='--')
        plt.savefig(save_path)
        plt.close()

if __name__ == "__main__":
    print("Running DuckDB Join Benchmark...")
    analyzer = DuckDBJoinBenchmark(data_size=100_000_000)
    results = analyzer.run_analysis()
    
    print("\nJoin Benchmark Results:")
    print(f"{'Metric':<10} | {'Mean (s)':<8} | {'Std Dev':<8}")
    print("-"*32)
    for metric, data in results.items():
        print(f"{metric:<10} | {data['mean']:.4f}   | {data['std']:.4f}")
    
    print("DuckDB Python module loaded from:", duckdb.__file__)


