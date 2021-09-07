from pathlib import Path
from typing import Dict, List, Tuple

DATA_DIR = Path(__file__).parent.parent / 'data'


def get_aen_luc_benchmark(
) -> Dict[Tuple[str, str], List[Tuple[Tuple[str, str], int]]]:
    results = {}
    benchmark_path = DATA_DIR / 'aen_luc1_hand.txt'
    with benchmark_path.open() as ifh:
        # skip header line
        next(ifh)
        for line in ifh:
            line = line.strip()
            if line:
                data = line.split('\t')
                lucan_locus = f'{data[0]}.{data[1]}'
                lucan_quote = data[2]
                vergil_locus = f'{data[3]}.{data[4]}'
                vergil_quote = data[5]
                rank = int(data[6])
                key = (vergil_locus, lucan_locus)
                if key in results:
                    results[key].append(((vergil_quote, lucan_quote), rank))
                else:
                    results[key] = [((vergil_quote, lucan_quote), rank)]
    return results
