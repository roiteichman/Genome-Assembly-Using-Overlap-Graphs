# Constants
lower_bound_l = 50
upper_bound_l = 150
lower_bound_n = 100
upper_bound_n = 1000000
lower_bound_p = 0.001
upper_bound_p = 0.1
metrics = ["Number of Contigs", "Genome Coverage", "N50", "Mismatch Rate Aligned Regions", "Mismatch Rate Genome Level"]
metric_labels = ["Number of Contigs", "Genome Coverage (%)", "N50", "Mismatch Rate Aligned Regions (%)",
                 "Mismatch Rate Genome (%)"]

def get_lower_bound_l():
    return lower_bound_l


def get_upper_bound_l():
    return upper_bound_l


def get_lower_bound_n():
    return lower_bound_n


def get_upper_bound_n():
    return upper_bound_n


def get_lower_bound_p():
    return lower_bound_p


def get_upper_bound_p():
    return upper_bound_p

def get_metrics():
    return metrics


def get_metric_labels():
    return metric_labels
