
import os
from dhnamlib.pylib import filesys
from splogic.utility.acceleration import accelerator


@accelerator.within_local_main_process
def save_analysis(analysis, dir_path):
    domain_analysis_dict = {}
    for example_analysis in analysis:
        domain_analysis_dict.setdefault(example_analysis['domain'], []).append(example_analysis)

    for domain, domain_analysis in domain_analysis_dict.items():
        domain_analysis_file_path = os.path.join(dir_path, f'analysis-{domain}.json')
        filesys.json_pretty_save(domain_analysis, domain_analysis_file_path)


@accelerator.within_local_main_process
def save_extra_performance(extra_performance, dir_path, file_name='extra_performance.json'):
    filesys.extended_json_pretty_save(extra_performance, os.path.join(dir_path, file_name))

    updated_performance = dict(extra_performance)
    for domain, performance in updated_performance.items():
        if 'accuracy' in performance:
            performance.update(accuracy_percent='{:5.2f}'.format(performance['accuracy'] * 100))
        if 'oracle_accuracy' in performance:
            performance.update(oracle_accuracy_percent='{:5.2f}'.format(performance['oracle_accuracy'] * 100))

    name, extension = os.path.splitext(file_name)
    new_file_name = f'{name}-visual{extension}'
    filesys.extended_json_pretty_save(updated_performance, os.path.join(dir_path, new_file_name))
