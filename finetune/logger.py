import torch

class Logger(object):
    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert len(result) == 2
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, run=None, type=None):
        if run is not None:
            result = torch.tensor(self.results[run])
            if type == 'cls':
                argmax = result[:, 0].argmax().item()
                print(f'Final results of Run {run + 1:02d}:')
                print(f'Best Epoch: {argmax+1:2d}')
                print(f'Best Valid AUC: {result[argmax, 0]:.6f}')
                print(f'Final Test AUC: {result[argmax, 1]:.6f}')
            else:
                argmin = result[:, 0].argmin().item()
                print(f'Final results of Run {run + 1:02d}:')
                print(f'Best Epoch: {argmin+1:2d}')
                print(f'Best Valid: {result[argmin, 0]:.6f}')
                print(f'Final Test: {result[argmin, 1]:.6f}')
        else:
            result = torch.tensor(self.results)

            best_results = []
            if type == 'cls':
                for r in result:
                    valid = r[:, 0].max().item()
                    test = r[r[:, 0].argmax(), 1].item()
                    best_results.append((valid, test))
            else:
                for r in result:
                    valid = r[:, 0].min().item()
                    test = r[r[:, 0].argmin(), 1].item()
                    best_results.append((valid, test))

            best_result = torch.tensor(best_results)

            print(f'All runs:')
            r = best_result[:, 0]
            print(f'  Best Valid: {r.mean():.6f} ± {r.std():.6f}')
            r = best_result[:, 1]
            print(f'  Final Test: {r.mean():.6f} ± {r.std():.6f}')
