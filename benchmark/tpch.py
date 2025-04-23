import duckdb
import timeit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class DuckDBAnalysis:
    def __init__(self, data_size=1e6):
        self.data_size = int(data_size)
        self.conn = duckdb.connect(':memory:')
        self._load_data()
        
    def _load_data(self):
        """Generate and load TPC-H-like data"""
        # Fact table
        lineitem = pd.DataFrame({
            'orderkey': np.arange(self.data_size),
            'totalprice': np.random.normal(100, 50, self.data_size),
            'partkey': np.random.randint(1, 2000, self.data_size),
            'quantity': np.random.randint(1, 50, self.data_size),
            'discount': np.random.uniform(0, 0.1, self.data_size)
        })
        
        # Get unique partkeys
        unique_partkeys = np.unique(lineitem['partkey'])
        num_parts = len(unique_partkeys)
        
        # Dimension table
        part = pd.DataFrame({
            'partkey': unique_partkeys,
             'name': pd.Categorical([f'Part_{i}' for i in range(num_parts)]),
            'retailprice': np.random.uniform(10, 500, num_parts)
        })
        
        self.conn.register('lineitem', lineitem)
        self.conn.register('part', part)
    
    def _benchmark(self, query, iterations=10):
        """Run benchmark with warmup"""
        times = []
        
        # Warmup
        for _ in range(2):
            self.conn.execute(query)
            
        # Timing
        for _ in range(iterations):
            start = timeit.default_timer()
            self.conn.execute(query).fetchall()
            times.append(timeit.default_timer() - start)
            
        return {
            'mean': np.mean(times),
            'std': np.std(times),
            'raw': times
        }
    
    def run_analysis(self):
        """Run all benchmarks"""
        results = {}
        
        # Join Benchmark
        join_query = """
            SELECT p.name, SUM(l.totalprice)
            FROM lineitem l
            JOIN part p ON l.partkey = p.partkey
            GROUP BY p.name
        """
        results['Join'] = self._benchmark(join_query)
        
        # Aggregation Benchmark  
        agg_query = """
            SELECT 
                partkey, 
                SUM(quantity * (1 - discount)),
                AVG(totalprice)
            FROM lineitem
            GROUP BY partkey
        """
        results['Aggregation'] = self._benchmark(agg_query)
        
        return results
    
    def visualize(self, results, save_path='benchmark_results.png'):
        """Create visualization"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        metrics = ['Join', 'Aggregation']
        means = [results[m]['mean'] for m in metrics]
        stds = [results[m]['std'] for m in metrics]
        
        ax.bar(metrics, means, yerr=stds, capsize=10)
        ax.set_title(f'DuckDB Standard Performance ({self.data_size:,} rows)')
        ax.set_ylabel('Execution Time (seconds)')
        ax.grid(axis='y', linestyle='--')
        
        plt.savefig(save_path)
        plt.close()

if __name__ == "__main__":
    print("Running DuckDB Standard Benchmark...")
    
    # Initialize with 1 million rows
    analyzer = DuckDBAnalysis(data_size=100_000_000)
    
    # Run benchmarks
    results = analyzer.run_analysis()
    
    # Print results
    print("\nBenchmark Results:")
    print(f"{'Metric':<12} | {'Mean (s)':<8} | {'Std Dev':<8}")
    print("-"*35)
    for metric, data in results.items():
        print(f"{metric:<12} | {data['mean']:.4f}   | {data['std']:.4f}")
    
    # Save visualization
    analyzer.visualize(results)
    print("\nVisualization saved to benchmark_results.png")