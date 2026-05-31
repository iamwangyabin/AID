import csv
import argparse
import hydra
import pickle

import torch
import torch.utils.data

import data
from utils.common import load_config_with_cli
from utils.evaluator import evaluate_model
from utils.network_factory import get_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--cfg', type=str, default=None, required=True)
    args, cfg_args = parser.parse_known_args()
    conf = load_config_with_cli(args.cfg, args_list=cfg_args)
    conf = hydra.utils.instantiate(conf)

    model = get_model(conf)
    resume_fn = hydra.utils.get_method(conf.resume.target)
    resume_fn(model, conf.resume.path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    all_results = []
    save_raw_results = {}

    for sub_data in conf.datasets.source:
        for sub_set in sub_data.sub_sets:
            dataset_cls = hydra.utils.get_class(sub_data.target)
            dataset = dataset_cls(sub_data.data_root, conf.datasets.trsf, subset=sub_set, split=sub_data.split)
            data_loader = torch.utils.data.DataLoader(dataset, batch_size=conf.datasets.batch_size,
                                                      num_workers=conf.datasets.loader_workers, shuffle=False)

            result = evaluate_model(model, data_loader, predictor=conf.eval_pipeline, device=device, desc=f"{sub_data.benchmark_name} {sub_set}")

            ap = result['ap']
            auc = result['auc']
            f1 = result['f1']
            r_acc0 = result['r_acc0']
            f_acc0 = result['f_acc0']
            acc0 = result['acc0']
            num_real = result['num_real']
            num_fake = result['num_fake']

            print(f"{sub_data.benchmark_name} {sub_set}")
            print(f"AP: {ap:.4f},\tF1: {f1:.4f},\tAUC: {auc:.4f},\tACC: {acc0:.4f},\tR_ACC: {r_acc0:.4f},\tF_ACC: {f_acc0:.4f}")
            all_results.append([sub_data.benchmark_name, sub_set, ap, auc, f1, r_acc0, f_acc0, acc0, num_real, num_fake])
            save_raw_results[f"{sub_data.benchmark_name} {sub_set}"] = result


    columns = ['dataset', 'sub_set', 'ap', 'auc', 'f1', 'r_acc0', 'f_acc0', 'acc0', 'num_real', 'num_fake']
    with open(conf.test_name+'_results.csv', 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(columns)
        for values in all_results:
            writer.writerow(values)
    with open(conf.test_name + '.pkl', 'wb') as file:
        pickle.dump(save_raw_results, file)
