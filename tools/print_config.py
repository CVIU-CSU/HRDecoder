# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications: Also save config as json file

import argparse

from mmcv import Config, DictAction

from mmseg.apis import init_segmentor


def parse_args():
    parser = argparse.ArgumentParser(description='Print the whole config')
    parser.add_argument('config', help='config file path')
    parser.add_argument(
        '--graph', action='store_true', help='print the models graph')
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='arguments in dict')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    print(f'Config:\n{cfg.pretty_text}')
    # dump config
    cfg.dump('example.py')
    super(Config, cfg).__setattr__('_filename', 'example.json')
    cfg.dump('example.json')
    # dump models graph
    if args.graph:
        model = init_segmentor(args.config, device='cpu')
        print(f'Model graph:\n{str(model)}')
        with open('example-graph.txt', 'w') as f:
            f.writelines(str(model))


if __name__ == '__main__':
    main()
