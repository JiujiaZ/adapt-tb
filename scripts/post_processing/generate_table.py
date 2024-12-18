import numpy as np
import pandas as pd


def find_exceeding_week(ref_mean, query_mean):
    # Create a mask where query is less than ref
    mask = query_mean < ref_mean

    # Find the first index where all subsequent values are True
    for i in range(len(mask)):
        if mask[i:].all():  # Checks if all values from index i to the end are True
            return i + 1  # Return 1-based index
    return -1  # Return -1 if no such week exists


def export_comparsion_summaries(K = 2, r = 2, d = 0.43, ref = 'historic TB rates', queries = ['exp3', 'LinUCB']):

    # methods = ['random', 'historic TB rates', 'exp3', 'LinUCB']

    # for each query, return exceeding week, week of peak, and peak improvement
    read_dir = 'data/output/simulation/'
    data = np.load(f'{read_dir}simulated_data_K{K}_r{r}_d{int(d * 100)}.npz') # ['method] [# repeats, # weeks, total, + ] cummulatively

    results = {
        'K': K,
        'r': r,
        'd': d
    }

    # ref_mean = (data[ref][:, :, 0] / data[ref][:, :, 1]).mean(axis = 0)
    ref_mean = np.where(data[ref][:, :, 1] == 0, np.nan, data[ref][:, :, 0] / data[ref][:, :, 1]).mean(axis=0)

    for query in queries:
        query_mean = (data[query][:, :, 0] / data[query][:, :, 1]).mean(axis = 0)
        # query_mean = np.where(data[query][:, :, 1] == 0, np.nan, data[query][:, :, 0] / data[query][:, :, 1]).mean(
        #     axis=0)

        # Compute exceeding_week
        exceeding_week = find_exceeding_week(ref_mean, query_mean)
        ratio = (ref_mean - query_mean) / ref_mean

        peak_week = np.nanargmax(ratio) + 1
        peak_percentage = int(round(ratio[peak_week - 1] * 100)) if not np.isnan(ratio[peak_week - 1]) else np.nan

        # peak_percentage = int(round(ratio[peak_week - 1] * 100))

        results[f'{query} exceeding_week'] = exceeding_week
        results[f'{query} peak_week'] = peak_week
        results[f'{query} peak_percentage'] = peak_percentage

    return results

def export_totals(K = 2, r = 2, d = 0.43):
    read_dir = 'data/output/simulation/'
    data = np.load(
        f'{read_dir}simulated_data_K{K}_r{r}_d{int(d * 100)}.npz')  # ['method] [# repeats, # weeks, total, + ] cummulatively

    results = {
        'K': K,
        'r': r,
        'd': d
    }

    for query in data.keys():
        query_data = data[query]

        query_total = query_data.sum(axis = 1)[:, 0]
        query_pos = query_data.sum(axis=1)[:, 1]

        results[f'{query} total_mean'] = round(int(query_total.mean()))
        results[f'{query} total_std'] = round(int(query_total.std()))
        results[f'{query} pos_mean'] = round(int(query_pos.mean()))
        results[f'{query} pos_std'] = round(int(query_pos.std()))

    return results

def export_yields(K = 2, r = 2, d = 0.43):
    read_dir = 'data/output/simulation/'
    data = np.load(
        f'{read_dir}simulated_data_K{K}_r{r}_d{int(d * 100)}.npz')  # ['method] [# repeats, # weeks, total, + ] cummulatively

    results = {
        'K': K,
        'r': r,
        'd': d
    }

    for query in data.keys():
        query_data = data[query]

        query_yield = query_data[:, -1, 0] / query_data[:, -1, 1]

        query_mean = query_yield.mean()
        query_std = query_yield.std()

        results[f'{query} yield_mean'] = round(int(query_mean))
        results[f'{query} yield_std'] = round(int(query_std))

    return results


def main(Ks = [1,2,3,4], rs = [1,2,3,4,5], ds = [0.23, 0.33, 0.43], queries = ['exp3', 'LinUCB']):
    results = []
    for K in Ks:
        for r in rs:
            for d in ds:
                result = export_comparsion_summaries(K, r, d, queries = queries)
                results.append(result)
    df = pd.DataFrame(results)

    save_dir = 'scripts/post_processing/'
    df.to_csv(save_dir + 'performance_benchmark.csv', index=False)

    results = []
    for K in Ks:
        for r in rs:
            for d in ds:
                result = export_totals(K, r, d)
                results.append(result)
    df = pd.DataFrame(results)

    save_dir = 'scripts/post_processing/'
    df.to_csv(save_dir + 'screening_summary.csv', index=False)

    results = []
    for K in Ks:
        for r in rs:
            for d in ds:
                result = export_yields(K, r, d)
                results.append(result)
    df = pd.DataFrame(results)
    save_dir = 'scripts/post_processing/'
    df.to_csv(save_dir + 'screening_yields.csv', index=False)

if __name__ == "__main__":
    main()


















